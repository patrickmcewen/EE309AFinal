#!/usr/bin/env python3
"""
Hybrid C++ DESTINY + Python DESTINY Workflow
==============================================

This script integrates:
1. C++ DESTINY - Performs design space exploration (DSE) to find optimal configuration
2. Python DESTINY - Computes symbolic expressions for memory access time using optimal config

Workflow:
  Input Config → C++ DESTINY (DSE) → Optimal Config → Python DESTINY (Symbolic) → Expressions
"""

import subprocess
import sys
import os
import argparse
import logging
from pathlib import Path
import datetime
import pyomo.environ as pyo
import sympy as sp
logger = logging.getLogger("hybrid_destiny_workflow")

sys.path.append(os.path.join(os.path.dirname(__file__), "destiny_3d_cache_python/scripts"))

import preprocess
from destiny_3d_cache_python.scripts.symbolic_access_time_FIXED import run_python_destiny_calculation
from destiny_3d_cache_python.scripts.symbolic_access_time_FIXED import compare_results
from destiny_3d_cache_python.scripts.symbolic_access_time_FIXED import compare_results_power    
from destiny_3d_cache_python.scripts.symbolic_access_time_FIXED import compute_sensitivity
from destiny_3d_cache_python.scripts.parse_cpp_output import parse_cpp_destiny_output, print_configuration


symbol_table = {}

DEBUG = True
def debug_print(message):
    if DEBUG:
        print(message)
    else:
        debug_print(message)

def init_symbol_table(param_values):
    for key in param_values:
        symbol_table[key.name] = key

def parse_ipopt_output(f, param_values):
    """
    Parses the output file from the optimizer in the inverse pass, mapping variable names to
    technology parameters and updating them accordingly.

    Args:
        f (file-like): Opened file object containing the output to parse.

    Returns:
        None
    """
    lines = f.readlines()
    mapping = {}
    max_ind = 0
    i = 0
    while lines[i][0] != "x":
        i += 1
    while lines[i][0] == "x":
        mapping[lines[i][lines[i].find("[") + 1 : lines[i].find("]")]] = (
            symbol_table[lines[i].split(" ")[-1][:-1]]
        )
        max_ind = int(lines[i][lines[i].find("[") + 1 : lines[i].find("]")])
        i += 1
    while i < len(lines) and lines[i].find("x") != 4:
        i += 1
    i += 2
    #print(f"mapping: {mapping}, max_ind: {max_ind}")
    for _ in range(max_ind):
        key = lines[i].split(":")[0].lstrip().rstrip()
        value = float(lines[i].split(":")[2][1:-1])
        if key in mapping:
            #print(f"key: {key}; mapping: {mapping[key]}; value: {value}")
            param_values[mapping[key]] = (
                value
            )
        i += 1

def run_cpp_destiny(config_file_name: str, output_file_name: str) -> int:
    """
    Run C++ DESTINY for design space exploration

    Args:
        config_file_name: Name of configuration file
        output_file_name: Name of output file
        cpp_destiny_path: Path to C++ DESTINY executable

    Returns:
        Return code (0 = success)
    """
    cpp_destiny_path = os.path.join(os.path.dirname(__file__), "destiny_3d_cache/destiny")
    config_folder = os.path.join(os.path.dirname(__file__), "destiny_3d_cache/config")
    config_file_path = os.path.join(config_folder, config_file_name)
    output_folder = os.path.join(os.path.dirname(__file__), "destiny_3d_cache/output")
    output_file_path = os.path.join(output_folder, output_file_name)
    # Convert to absolute path before changing directories
    output_file_path = os.path.abspath(output_file_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    old_cwd = os.getcwd()
    debug_print(f"=" * 80)
    debug_print(f"STEP 1: Running C++ DESTINY for Design Space Exploration")
    debug_print(f"=" * 80)
    debug_print(f"\nConfiguration: {config_file_name}")
    debug_print(f"Output file: {output_file_name}")

    os.chdir(config_folder)

    # DESTINY executable doesn't support -o flag and only writes to file in full_exploration mode
    # For other modes, it prints to stdout, so we capture stdout and write to file
    # We use Popen to both display output in real-time and save it to file
    cmd = [cpp_destiny_path, config_file_path]

    try:
        debug_print(f"Running command: {cmd} from {os.getcwd()}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Write output to file while also displaying it
        with open(output_file_path, 'w') as f:
            for line in process.stdout:
                debug_print(f"{line}")  # Display in real-time
                f.write(line)  # Save to file
        
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
        debug_print(f"\n✓ C++ DESTINY DSE completed successfully!")
        debug_print(f"  Optimal configuration saved to: {output_file_path}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ C++ DESTINY failed with error code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"\n✗ C++ DESTINY executable not found: {cpp_destiny_path}")
        return 1
    finally:
        os.chdir(old_cwd)


def run_python_symbolic_analysis(cpp_output_file_name: str, config_file_name: str, python_script: str) -> int:
    """
    Run Python DESTINY for symbolic access time analysis

    Args:
        cpp_output_file_name: C++ DESTINY output file name with optimal configuration
        config_file_name: Original configuration file name
        python_script: Path to symbolic analysis Python script

    Returns:
        Return code (0 = success)
    """
    destiny_python_path = os.path.join(os.path.dirname(__file__), "destiny_3d_cache_python")
    python_script_path = os.path.join(os.path.dirname(__file__), "destiny_3d_cache_python/scripts", python_script)
    config_folder = os.path.join(os.path.dirname(__file__), "destiny_3d_cache/config")
    config_file_path = os.path.join(config_folder, config_file_name)
    cpp_output_folder = os.path.join(os.path.dirname(__file__), "destiny_3d_cache/output")
    cpp_output_file_path = os.path.join(cpp_output_folder, cpp_output_file_name)
    old_cwd = os.getcwd()
    debug_print(f"\n" + "=" * 80)
    debug_print(f"STEP 2: Running Python DESTINY for Symbolic Analysis")
    debug_print(f"=" * 80)
    debug_print(f"\nUsing optimal configuration from: {cpp_output_file_path}")
    debug_print(f"Computing symbolic expressions for memory access time...")

    os.chdir(destiny_python_path)

    

    opt_config = parse_cpp_destiny_output(cpp_output_file_path)
    print_configuration(opt_config)
    subarray, bank = run_python_destiny_calculation(opt_config, config_file_path)
    if type(bank.readLatency) == float:
        print(f"bank.readLatency is a float: {bank.readLatency}")
    else:
        print(f"bank.readLatency type: {type(bank.readLatency.symbolic)}")
    compare_results(subarray, bank, opt_config)
    compare_results_power(subarray, bank, opt_config)
    compute_sensitivity(subarray, bank, opt_config)
    init_symbol_table(bank.readLatency.val_map)
    constraints = init_constraints(bank, bank.readLatency, 1.5)
    return run_ipopt_optimization(bank.readLatency.symbolic, constraints, bank.readLatency.val_map, bank)

def init_constraints(bank, obj, improvement):
    """
    Initialize constraints for IPOPT optimization
    """
    disabled_knobs = [
        "barrierThickness_localWire",
        "effectiveResistanceMultiplier_tech_peripheral",
        "mD_fac_tech_peripheral",
        "effectiveResistanceMultiplier_cell_tech",
        "mD_fac_cell_tech",
        "barrierThickness_globalWire",
    ]
    match_pattern_disable = [
        "cell_tech",
        "localWire"
    ]
    constraints = []
    constraints.append(obj.symbolic >= obj.concrete / improvement)
    for knob in disabled_knobs:
        if knob in symbol_table:
            constraints.append(sp.Eq(symbol_table[knob], bank.readLatency.val_map[symbol_table[knob]], evaluate=False))
        else:
            print(f"{knob} not found in symbol_table")
    for pattern in match_pattern_disable:
        for param in symbol_table:
            if pattern in param:
                constraints.append(sp.Eq(symbol_table[param], bank.readLatency.val_map[symbol_table[param]], evaluate=False))
    constraints.append(symbol_table["vth_tech_peripheral"] <= symbol_table["vdd_tech_peripheral"])
    #print(f"currentOffNmos[0]: {bank.g.tech.currentOffNmos[0].concrete}")
    #print(f"currentOffPmos[0]: {bank.g.tech.currentOffPmos[0].concrete}")
    #constraints.append(bank.g.tech.currentOffNmos[0].symbolic <= (100e-9 / (1e-6))) # leakage <= 100nA/um
    #constraints.append(bank.g.tech.currentOffPmos[0].symbolic <= (100e-9 / (1e-6))) # leakage <= 100nA/um
    constraints.append(bank.g.tech.currentOffNmos[0].symbolic <= bank.g.tech.currentOffNmos[0].concrete)
    constraints.append(bank.g.tech.currentOffPmos[0].symbolic <= bank.g.tech.currentOffPmos[0].concrete) # no increase in leakage
    
    return constraints

def run_ipopt_optimization(obj, constraints, initial_values, bank) -> int:
    """
    Run IPOPT optimization
    """
    debug_print(f"\n" + "=" * 80)
    debug_print(f"STEP 3: Running IPOPT Optimization")
    debug_print(f"=" * 80)
    ipopt_path = os.path.join(os.path.dirname(__file__), "ipopt_out.txt")
    stdout = sys.stdout
    with open(ipopt_path, 'w') as sys.stdout:
        model = pyo.ConcreteModel()
        model_out_file = os.path.join(os.path.dirname(__file__), "solver_out.txt")
        preprocessor = preprocess.Preprocessor(initial_values, out_file=model_out_file)
        opt, scaled_model, model, multistart_options = preprocessor.begin(model, obj, False, constraints)
        results = opt.solve(scaled_model, tee=True)
        if results.solver.termination_condition not in ["optimal", "acceptable", "infeasible", "maxIterations"]:
            raise Exception(f"IPOPT optimization failed with termination condition: {results.solver.termination_condition}")
        pyo.TransformationFactory("core.scale_model").propagate_solution(scaled_model, model)
        model.display()
    sys.stdout = stdout
    prev_initial_values = initial_values.copy()
    with open(ipopt_path, 'r') as f:
        parse_ipopt_output(f, initial_values)
    
    debug_print(f"\nprinting values that changed during optimization:\n")
    for key in initial_values:
        if prev_initial_values[key] != 0:
            percent_change = abs(initial_values[key] - prev_initial_values[key]) / prev_initial_values[key]
        else:
            percent_change = abs(initial_values[key] - prev_initial_values[key])
        if percent_change > 0.01:
            debug_print(f"{key}: {prev_initial_values[key]} -> {initial_values[key]}")

    debug_print(f"\nprinting non-base values that changed during optimization:\n")
    v_th_eff_init = bank.g.tech.V_th_eff.symbolic.xreplace(prev_initial_values)
    v_th_eff_final = bank.g.tech.V_th_eff.symbolic.xreplace(initial_values)
    percent_change = abs(v_th_eff_final - v_th_eff_init) / v_th_eff_init
    if percent_change > 0.01:
        debug_print(f"v_th_eff: {v_th_eff_init} -> {v_th_eff_final}")
    current_off_nmos_init = bank.g.tech.currentOffNmos[0].symbolic.xreplace(prev_initial_values)
    current_off_nmos_final = bank.g.tech.currentOffNmos[0].symbolic.xreplace(initial_values)
    debug_print(f"current_off_nmos (uA/um): {current_off_nmos_init} -> {current_off_nmos_final}")
    current_off_pmos_init = bank.g.tech.currentOffPmos[0].symbolic.xreplace(prev_initial_values)
    current_off_pmos_final = bank.g.tech.currentOffPmos[0].symbolic.xreplace(initial_values)
    debug_print(f"current_off_pmos (uA/um): {current_off_pmos_init} -> {current_off_pmos_final}")

    current_on_nmos_init = bank.g.tech.currentOnNmos[0].symbolic.xreplace(prev_initial_values)
    current_on_nmos_final = bank.g.tech.currentOnNmos[0].symbolic.xreplace(initial_values)
    debug_print(f"current_on_nmos (uA/um): {current_on_nmos_init} -> {current_on_nmos_final}")
    current_on_pmos_init = bank.g.tech.currentOnPmos[0].symbolic.xreplace(prev_initial_values)
    current_on_pmos_final = bank.g.tech.currentOnPmos[0].symbolic.xreplace(initial_values)
    debug_print(f"current_on_pmos (uA/um): {current_on_pmos_init} -> {current_on_pmos_final}")

    read_latency_init = bank.readLatency.symbolic.xreplace(prev_initial_values)
    read_latency_final = bank.readLatency.symbolic.xreplace(initial_values)
    percent_change = abs(read_latency_final - read_latency_init) / read_latency_init
    if percent_change > 0.01:
        debug_print(f"read_latency (s): {float(read_latency_init)} -> {float(read_latency_final)}")
    return 0

def main():
    """Main workflow orchestrator"""
    parser = argparse.ArgumentParser(
        description="Hybrid C++ DESTINY + Python DESTINY Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample SRAM configuration
  python hybrid_destiny_workflow.py -c config/sample_SRAM_2layer.cfg

  # Specify custom paths
  python hybrid_destiny_workflow.py -c my_config.cfg \\
      --cpp-destiny destiny_3d_cache-master/destiny \\
      --output results/my_output.txt

Notes:
  - C++ DESTINY performs design space exploration (may take several minutes)
  - Python DESTINY uses the optimal configuration for symbolic analysis
  - All symbolic expressions and sensitivity analysis are printed to console
        """
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save logs (default: logs)"
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Configuration file name (e.g., sample_SRAM_2layer.cfg)",
        default="sample_SRAM_2layer.cfg",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="hybrid_output.txt",
        help="Output file for C++ DESTINY results (default: hybrid_output.txt)"
    )

    parser.add_argument(
        "--python-script",
        type=str,
        default="symbolic_access_time_FIXED.py",
        help="Name of Python symbolic analysis script"
    )

    parser.add_argument(
        "--skip-cpp",
        action="store_true",
        help="Skip C++ DESTINY and use existing output file"
    )

    parser.add_argument(
        "--symbolic-tech-model",
        action="store_true",
        help="Use symbolic technology model",
        default=True
    )

    args = parser.parse_args()

    this_run_log_dir = os.path.join(os.path.dirname(__file__), "..", args.log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(this_run_log_dir):
        os.makedirs(this_run_log_dir)
    logging.basicConfig(filename=os.path.join(this_run_log_dir, "hybrid_destiny_workflow.log"), level=logging.INFO)

    debug_print(f"\n")
    debug_print(f"=" * 80)
    debug_print(f"| " + " " * 15 + "Hybrid C++ DESTINY + Python DESTINY Workflow" + " " * 18 + "|")
    debug_print(f"=" * 80)

    debug_print(f"\nWorkflow:")
    debug_print(f"  1. C++ DESTINY performs Design Space Exploration (DSE)")
    debug_print(f"  2. Extracts optimal configuration from DSE results")
    debug_print(f"  3. Python DESTINY computes symbolic expressions for access time")
    debug_print(f"  4. Performs sensitivity analysis on symbolic expressions")

    output_file_name = args.output.replace(".txt", f"_{args.config.replace('.cfg', '')}.txt")

    # Step 1: Run C++ DESTINY (unless skipped)
    if not args.skip_cpp:
        ret = run_cpp_destiny(args.config, output_file_name)
        if ret != 0:
            debug_print(f"\nWorkflow failed at C++ DESTINY stage")
            return ret
    else:
        debug_print(f"\nSkipping C++ DESTINY (using existing output)")
        output_file_path = os.path.join(os.path.dirname(__file__), "destiny_3d_cache/output", output_file_name)
        if not os.path.exists(output_file_path):
            debug_print(f"Output file not found: {output_file_path}")
            return 1

    # Step 2: Run Python DESTINY symbolic analysis
    ret = run_python_symbolic_analysis(output_file_name, args.config, args.python_script)
    if ret != 0:
        debug_print(f"\nWorkflow failed at Python DESTINY stage")
        return ret

    # Success!
    debug_print(f"\n" + "=" * 80)
    debug_print(f"HYBRID WORKFLOW COMPLETED SUCCESSFULLY!")
    debug_print(f"=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
