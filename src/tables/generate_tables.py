#!/usr/bin/env python3
"""
Script to generate LaTeX tables from workflow output files.

This script parses workflow output files and generates three types of tables:
1. Configuration table - Architecture configurations
2. Sensitivity table - Normalized sensitivity values
3. Validation table - Timing comparison between Python and C++

Usage:
    python generate_tables.py --output-dir <output_dir> [--files <file1> <file2> ...]
"""

import re
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class WorkflowOutputParser:
    """Parser for workflow output files"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.tech_name = self._extract_tech_name(file_path)
        with open(file_path, 'r') as f:
            self.content = f.read()
        
        self.config = self._parse_configuration()
        self.timing = self._parse_timing()
        self.sensitivity = self._parse_sensitivity()
        self.optimization = self._parse_optimization()
    
    def _extract_tech_name(self, file_path: str) -> str:
        """Extract technology name from filename"""
        filename = os.path.basename(file_path)
        # Remove 'sample_' prefix and '_out.txt' suffix
        name = filename.replace('sample_', '').replace('_out.txt', '')
        # Handle layer suffixes first
        if '_2layer' in name:
            name = name.replace('_2layer', '')
        if '_4layer' in name:
            name = name.replace('_4layer', '')
        
        # Format common names
        if 'SRAM' in name:
            return 'SRAM'
        elif 'STTRAM' in name or 'STT-RAM' in name:
            return 'STT-RAM'
        elif '2D' in name and 'eDRAM' in name:
            return '2D eDRAM'
        elif '2D' in name and 'ReRAM' in name:
            return '2D ReRAM'
        elif 'PCRAM' in name:
            return 'PCRAM'
        
        # Format nicely for other cases
        if '2D' in name:
            name = name.replace('2D', '2D ')
        # Clean up any remaining underscores and spaces
        name = name.replace('_', ' ').replace('  ', ' ')
        return name.strip()
    
    def _parse_configuration(self) -> Dict:
        """Parse configuration section"""
        config = {}
        
        # Bank Organization
        bank_match = re.search(r'Banks \(X x Y x Stacks\):\s*(\d+)\s*x\s*(\d+)(?:\s*x\s*(\d+))?', self.content)
        if bank_match:
            config['bank_x'] = int(bank_match.group(1))
            config['bank_y'] = int(bank_match.group(2))
            config['bank_stacks'] = int(bank_match.group(3)) if bank_match.group(3) else None
        
        # Row/Column Activation (Bank level) - from HIERARCHY SUMMARY section
        hierarchy_match = re.search(r'HIERARCHY SUMMARY.*?Bank Organization.*?Row Activation\s*:\s*(\d+)\s*/\s*(\d+)', self.content, re.DOTALL)
        if hierarchy_match:
            config['bank_row_act'] = f"{hierarchy_match.group(1)} / {hierarchy_match.group(2)}"
        
        hierarchy_col_match = re.search(r'HIERARCHY SUMMARY.*?Column Activation\s*:\s*(\d+)\s*/\s*(\d+)', self.content, re.DOTALL)
        if hierarchy_col_match:
            config['bank_col_act'] = f"{hierarchy_col_match.group(1)} / {hierarchy_col_match.group(2)}"
        
        # Also try from Bank Organization section if not found
        if 'bank_row_act' not in config:
            bank_row_act = re.search(r'Bank Organization.*?Row Activation\s*:\s*(\d+)\s*/\s*(\d+)', self.content, re.DOTALL)
            if bank_row_act:
                config['bank_row_act'] = f"{bank_row_act.group(1)} / {bank_row_act.group(2)}"
        
        if 'bank_col_act' not in config:
            bank_col_act = re.search(r'Bank Organization.*?Column Activation\s*:\s*(\d+)\s*/\s*(\d+)', self.content, re.DOTALL)
            if bank_col_act:
                config['bank_col_act'] = f"{bank_col_act.group(1)} / {bank_col_act.group(2)}"
        
        # Mat Organization
        mat_match = re.search(r'Mats \(X x Y\):\s*(\d+)\s*x\s*(\d+)', self.content)
        if mat_match:
            config['mat_x'] = int(mat_match.group(1))
            config['mat_y'] = int(mat_match.group(2))
        
        # Mat Row/Column Activation - from HIERARCHY SUMMARY section
        mat_row_act = re.search(r'HIERARCHY SUMMARY.*?Mat Organization.*?Row Activation\s*:\s*(\d+)\s*/\s*(\d+)', self.content, re.DOTALL)
        if mat_row_act:
            config['mat_row_act'] = f"{mat_row_act.group(1)} / {mat_row_act.group(2)}"
        
        mat_col_act = re.search(r'HIERARCHY SUMMARY.*?Mat Organization.*?Column Activation\s*:\s*(\d+)\s*/\s*(\d+)', self.content, re.DOTALL)
        if mat_col_act:
            config['mat_col_act'] = f"{mat_col_act.group(1)} / {mat_col_act.group(2)}"
        
        # Also try from Mat Organization section if not found
        if 'mat_row_act' not in config:
            mat_row_act = re.search(r'Mat Organization.*?Row Activation\s*:\s*(\d+)\s*/\s*(\d+)', self.content, re.DOTALL)
            if mat_row_act:
                config['mat_row_act'] = f"{mat_row_act.group(1)} / {mat_row_act.group(2)}"
        
        if 'mat_col_act' not in config:
            mat_col_act = re.search(r'Mat Organization.*?Column Activation\s*:\s*(\d+)\s*/\s*(\d+)', self.content, re.DOTALL)
            if mat_col_act:
                config['mat_col_act'] = f"{mat_col_act.group(1)} / {mat_col_act.group(2)}"
        
        # Subarray Size
        subarray_match = re.search(r'Size:\s*(\d+)\s*Rows\s*x\s*(\d+)\s*Columns', self.content)
        if subarray_match:
            config['subarray_rows'] = int(subarray_match.group(1))
            config['subarray_cols'] = int(subarray_match.group(2))
        
        # Mux Levels
        senseamp_mux = re.search(r'Senseamp Mux:\s*(\d+)', self.content)
        if senseamp_mux:
            config['senseamp_mux'] = int(senseamp_mux.group(1))
        
        output_l1 = re.search(r'Output L1 Mux:\s*(\d+)', self.content)
        if output_l1:
            config['output_l1_mux'] = int(output_l1.group(1))
        
        output_l2 = re.search(r'Output L2 Mux:\s*(\d+)', self.content)
        if output_l2:
            config['output_l2_mux'] = int(output_l2.group(1))
        
        # Rows per Set
        rows_per_set = re.search(r'Rows per Set\s*:\s*(\d+)', self.content)
        if rows_per_set:
            config['rows_per_set'] = int(rows_per_set.group(1))
        else:
            config['rows_per_set'] = 1  # Default
        
        # Wire types - try to find from C++ output or use defaults
        config['local_wire_type'] = 'Local Aggressive'
        config['global_wire_type'] = 'Global Aggressive'
        config['repeaters'] = 'None'
        config['low_swing'] = 'No'
        config['buffer_style'] = 'Latency-Optimized'
        
        return config
    
    def _parse_timing(self) -> Dict:
        """Parse timing section - both Python and C++"""
        timing = {}
        
        # C++ Timing
        cpp_read = re.search(r'Total Read Latency:\s*([\d.]+)\s*ns', self.content)
        if cpp_read:
            timing['cpp_read'] = float(cpp_read.group(1))
        
        cpp_write = re.search(r'Total Write Latency:\s*([\d.]+)\s*ns', self.content)
        if cpp_write:
            timing['cpp_write'] = float(cpp_write.group(1))
        
        cpp_htree = re.search(r'H-Tree Latency:\s*([\d.]+)\s*(ps|ns)', self.content)
        if cpp_htree:
            val = float(cpp_htree.group(1))
            unit = cpp_htree.group(2)
            timing['cpp_htree'] = val * (1e-3 if unit == 'ps' else 1.0)  # Convert to ns
        
        cpp_mat = re.search(r'Mat Latency:\s*([\d.]+)\s*ns', self.content)
        if cpp_mat:
            timing['cpp_mat'] = float(cpp_mat.group(1))
        
        cpp_predecoder = re.search(r'Predecoder:\s*([\d.]+)\s*(ps|ns)', self.content)
        if cpp_predecoder:
            val = float(cpp_predecoder.group(1))
            unit = cpp_predecoder.group(2)
            timing['cpp_predecoder'] = val * (1e-3 if unit == 'ps' else 1.0)  # Convert to ns
        
        cpp_subarray = re.search(r'Subarray:\s*([\d.]+)\s*ns', self.content)
        if cpp_subarray:
            timing['cpp_subarray'] = float(cpp_subarray.group(1))
        
        cpp_row_decoder = re.search(r'Row Decoder:\s*([\d.]+)\s*ns', self.content)
        if cpp_row_decoder:
            timing['cpp_row_decoder'] = float(cpp_row_decoder.group(1))
        
        cpp_bitline = re.search(r'Bitline:\s*([\d.]+)\s*(ps|ns)', self.content)
        if cpp_bitline:
            val = float(cpp_bitline.group(1))
            unit = cpp_bitline.group(2)
            timing['cpp_bitline'] = val * (1e-3 if unit == 'ps' else 1.0)  # Convert to ns
        
        cpp_senseamp = re.search(r'Senseamp:\s*([\d.]+)\s*(ps|ns)', self.content)
        if cpp_senseamp:
            val = float(cpp_senseamp.group(1))
            unit = cpp_senseamp.group(2)
            timing['cpp_senseamp'] = val * (1e-3 if unit == 'ps' else 1.0)  # Convert to ns
        
        cpp_mux = re.search(r'Mux:\s*([\d.]+)\s*(ps|ns)', self.content)
        if cpp_mux:
            val = float(cpp_mux.group(1))
            unit = cpp_mux.group(2)
            timing['cpp_mux'] = val * (1e-3 if unit == 'ps' else 1.0)  # Convert to ns
        
        # Python Timing (from comparison section)
        # Try multiple patterns
        py_read_match = re.search(r'Total Read Latency\s*:.*?([\d.]+)\s*ns\s*\(Py\)', self.content)
        if not py_read_match:
            # Try alternative pattern
            py_read_match = re.search(r'Total Read Latency\s*:.*?\(Py\)\s*:\s*([\d.]+)\s*ns', self.content)
        if py_read_match:
            timing['py_read'] = float(py_read_match.group(1))
        
        py_write_match = re.search(r'Total Write Latency\s*:.*?([\d.]+)\s*ns\s*\(Py\)', self.content)
        if not py_write_match:
            # Try alternative pattern
            py_write_match = re.search(r'Total Write Latency\s*:.*?\(Py\)\s*:\s*([\d.]+)\s*ns', self.content)
        if py_write_match:
            timing['py_write'] = float(py_write_match.group(1))
        
        py_htree_match = re.search(r'(?:H-Tree|Non-H-Tree) Latency\s+.*?([\d.]+)\s*(ps|ns)\s*\(Py\)', self.content)
        if py_htree_match:
            val = float(py_htree_match.group(1))
            unit = py_htree_match.group(2)
            timing['py_htree'] = val * (1e-3 if unit == 'ps' else 1.0)  # Convert to ns
        
        py_mat_match = re.search(r'Mat Latency\s+.*?([\d.]+)\s*ns\s*\(Py\)', self.content)
        if py_mat_match:
            timing['py_mat'] = float(py_mat_match.group(1))
        
        py_predecoder_match = re.search(r'Predecoder\s+.*?([\d.]+)\s*(ps|ns)\s*\(Py\)', self.content)
        if py_predecoder_match:
            val = float(py_predecoder_match.group(1))
            unit = py_predecoder_match.group(2)
            timing['py_predecoder'] = val * (1e-3 if unit == 'ps' else 1.0)
        
        py_subarray_match = re.search(r'Subarray\s+.*?([\d.]+)\s*ns\s*\(Py\)', self.content)
        if py_subarray_match:
            timing['py_subarray'] = float(py_subarray_match.group(1))
        
        py_row_decoder_match = re.search(r'Row Decoder\s+.*?([\d.]+)\s*ns\s*\(Py\)', self.content)
        if py_row_decoder_match:
            timing['py_row_decoder'] = float(py_row_decoder_match.group(1))
        
        py_bitline_match = re.search(r'Bitline\s+.*?([\d.]+)\s*(ps|ns)\s*\(Py\)', self.content)
        if py_bitline_match:
            val = float(py_bitline_match.group(1))
            unit = py_bitline_match.group(2)
            timing['py_bitline'] = val * (1e-3 if unit == 'ps' else 1.0)
        
        py_senseamp_match = re.search(r'Sense Amp\s+.*?([\d.]+)\s*(ps|ns)\s*\(Py\)', self.content)
        if py_senseamp_match:
            val = float(py_senseamp_match.group(1))
            unit = py_senseamp_match.group(2)
            timing['py_senseamp'] = val * (1e-3 if unit == 'ps' else 1.0)
        
        py_mux_match = re.search(r'Mux\s+.*?([\d.]+)\s*(ps|ns)\s*\(Py\)', self.content)
        if py_mux_match:
            val = float(py_mux_match.group(1))
            unit = py_mux_match.group(2)
            timing['py_mux'] = val * (1e-3 if unit == 'ps' else 1.0)
        
        return timing
    
    def _parse_sensitivity(self) -> Dict[str, Dict[str, float]]:
        """Parse sensitivity section"""
        sensitivity = defaultdict(dict)
        
        # Find read sensitivity section
        read_section = re.search(r'top read sensitivities.*?(?=now with write|$)', self.content, re.DOTALL)
        if read_section:
            for match in re.finditer(r'(\w+):\s*([+-]?[\d.e-]+)', read_section.group(0)):
                param = match.group(1)
                value = float(match.group(2))
                sensitivity[param]['read'] = value
        
        # Find write sensitivity section
        write_section = re.search(r'top write sensitivities.*?(?=currentOffNmos|$)', self.content, re.DOTALL)
        if write_section:
            for match in re.finditer(r'(\w+):\s*([+-]?[\d.e-]+)', write_section.group(0)):
                param = match.group(1)
                value = float(match.group(2))
                sensitivity[param]['write'] = value
        
        return dict(sensitivity)
    
    def _parse_optimization(self) -> Dict[str, Dict]:
        """Parse optimization results section"""
        optimization = {
            'base_params': {},
            'non_base_params': {}
        }
        
        # Find the optimization section
        opt_section = re.search(r'STEP 3: Running IPOPT Optimization.*?(?=HYBRID WORKFLOW|$)', self.content, re.DOTALL)
        if not opt_section:
            return optimization
        
        opt_content = opt_section.group(0)
        
        # Parse base parameters (tunable knobs)
        base_section = re.search(r'printing values that changed during optimization:(.*?)(?=printing non-base|$)', opt_content, re.DOTALL)
        if base_section:
            for line in base_section.group(1).strip().split('\n'):
                if not line.strip():
                    continue
                # Pattern: param: before -> after
                # Number pattern: optional sign, digits, optional decimal, optional scientific notation
                match = re.match(r'(\w+(?:_\w+)*):\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*->\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', line)
                if match:
                    param = match.group(1)
                    try:
                        before = float(match.group(2))
                        after = float(match.group(3))
                        optimization['base_params'][param] = {
                            'before': before,
                            'after': after
                        }
                    except ValueError:
                        continue
        
        # Parse non-base parameters (derived values)
        non_base_section = re.search(r'printing non-base values that changed during optimization:(.*?)(?=HYBRID WORKFLOW|$)', opt_content, re.DOTALL)
        if non_base_section:
            for line in non_base_section.group(1).strip().split('\n'):
                if not line.strip():
                    continue
                # Pattern: param (unit): before -> after
                # Number pattern: optional sign, digits, optional decimal, optional scientific notation
                match = re.match(r'(\w+(?:_\w+)*)\s*\(([^)]+)\):\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*->\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', line)
                if match:
                    param = match.group(1)
                    unit = match.group(2)
                    try:
                        before = float(match.group(3))
                        after = float(match.group(4))
                        optimization['non_base_params'][param] = {
                            'before': before,
                            'after': after,
                            'unit': unit
                        }
                    except ValueError:
                        continue
        
        return optimization


class LaTeXTableGenerator:
    """Generate LaTeX tables from parsed data"""
    
    @staticmethod
    def format_scientific(value: float, precision: int = 3) -> str:
        """Format number in scientific notation for LaTeX"""
        if abs(value) < 1e-10:
            return "$0$"
        
        # Convert to scientific notation
        exp = int(f"{value:.{precision}e}".split('e')[1])
        mantissa = value / (10 ** exp)
        
        return f"${mantissa:.{precision}f}\\times10^{{{exp}}}$"
    
    @staticmethod
    def generate_configuration_table(parsers: List[WorkflowOutputParser], output_file: str):
        """Generate configuration table"""
        with open(output_file, 'w') as f:
            f.write("\\begin{table*}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Resulting Architecture Configurations for ")
            tech_names = [p.tech_name for p in parsers]
            f.write(", ".join(tech_names))
            f.write(" decided by C++ DESTINY}\n")
            f.write("\\label{tab:configs_clean}\n")
            f.write("\\renewcommand{\\arraystretch}{1.25}\n\n")
            
            # Table columns
            num_cols = len(parsers)
            f.write("\\begin{tabular}{l")
            for _ in range(num_cols):
                f.write(" p{0.22\\textwidth}")
            f.write("}\n")
            f.write("\\toprule\n")
            
            # Header
            f.write("\\textbf{Configuration Field}")
            for parser in parsers:
                f.write(f" & \\centering \\textbf{{{parser.tech_name}}}")
            f.write("\\tabularnewline\n")
            f.write("\\midrule\n\n")
            
            # Bank Organization
            f.write("\\textbf{Bank Organization}\n")
            for parser in parsers:
                c = parser.config
                if c.get('bank_stacks'):
                    bank_str = f"{c['bank_x']} × {c['bank_y']} × {c['bank_stacks']}"
                else:
                    bank_str = f"{c['bank_x']} × {c['bank_y']}"
                f.write(f" & \\centering {bank_str}")
            f.write("\\tabularnewline\n\n")
            
            # Row Activation
            f.write("Row Activation\n")
            for parser in parsers:
                c = parser.config
                row_act = c.get('bank_row_act', 'N/A')
                f.write(f" & \\centering {row_act}")
            f.write("\\tabularnewline\n\n")
            
            # Column Activation
            f.write("Column Activation\n")
            for parser in parsers:
                c = parser.config
                col_act = c.get('bank_col_act', 'N/A')
                f.write(f" & \\centering {col_act}")
            f.write("\\tabularnewline\n\n")
            
            # Mat Organization
            f.write("\\textbf{Mat Organization}\n")
            for parser in parsers:
                c = parser.config
                mat_str = f"{c.get('mat_x', 'N/A')} × {c.get('mat_y', 'N/A')}"
                f.write(f" & \\centering {mat_str}")
            f.write("\\tabularnewline\n\n")
            
            # Mat Row Activation
            f.write("Mat Row Activation\n")
            for parser in parsers:
                c = parser.config
                mat_row_act = c.get('mat_row_act', 'N/A')
                f.write(f" & \\centering {mat_row_act}")
            f.write("\\tabularnewline\n\n")
            
            # Mat Column Activation
            f.write("Mat Column Activation\n")
            for parser in parsers:
                c = parser.config
                mat_col_act = c.get('mat_col_act', 'N/A')
                f.write(f" & \\centering {mat_col_act}")
            f.write("\\tabularnewline\n\n")
            
            # Subarray Size
            f.write("Subarray Size\n")
            for parser in parsers:
                c = parser.config
                subarray_str = f"{c.get('subarray_rows', 'N/A')} rows × {c.get('subarray_cols', 'N/A')} cols"
                f.write(f" & \\centering {subarray_str}")
            f.write("\\tabularnewline\n\n")
            
            # Mux Levels
            f.write("\\textbf{Mux Levels}\n")
            for parser in parsers:
                c = parser.config
                mux_str = f"Senseamp: {c.get('senseamp_mux', 'N/A')} \\\\ L1: {c.get('output_l1_mux', 'N/A')} \\\\ L2: {c.get('output_l2_mux', 'N/A')}"
                f.write(f" & \\centering {mux_str}")
            f.write("\\tabularnewline\n\n")
            
            # Set Partitioning
            f.write("Set Partitioning\n")
            for parser in parsers:
                c = parser.config
                rows_per_set = c.get('rows_per_set', 1)
                f.write(f" & \\centering {rows_per_set} row")
            f.write("\\tabularnewline\n\n")
            
            # Local Wire
            f.write("\\textbf{Local Wire}\n")
            for parser in parsers:
                c = parser.config
                wire_str = f"Type: {c.get('local_wire_type', 'N/A')} \\\\ Repeaters: {c.get('repeaters', 'N/A')} \\\\ Low Swing: {c.get('low_swing', 'N/A')}"
                f.write(f" & \\centering {wire_str}")
            f.write("\\tabularnewline\n\n")
            
            # Global Wire
            f.write("\\textbf{Global Wire}\n")
            for parser in parsers:
                c = parser.config
                wire_str = f"Type: {c.get('global_wire_type', 'N/A')} \\\\ Repeaters: {c.get('repeaters', 'N/A')} \\\\ Low Swing: {c.get('low_swing', 'N/A')}"
                f.write(f" & \\centering {wire_str}")
            f.write("\\tabularnewline\n\n")
            
            # Buffer Design Style
            f.write("\\textbf{Buffer Design Style}\n")
            for parser in parsers:
                c = parser.config
                buffer_style = c.get('buffer_style', 'N/A')
                f.write(f" & \\centering {buffer_style}")
            f.write("\\tabularnewline\n\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table*}\n")
    
    @staticmethod
    def generate_sensitivity_table(parsers: List[WorkflowOutputParser], output_file: str):
        """Generate sensitivity table"""
        # Find parameters that appear in ALL parsers (intersection)
        if not parsers:
            common_params = set()
        else:
            common_params = set(parsers[0].sensitivity.keys())
            for parser in parsers[1:]:
                common_params &= set(parser.sensitivity.keys())
        
        # Calculate average sensitivity for each common parameter
        # Average is computed as mean of absolute values of read and write sensitivities across all parsers
        param_avg_sensitivity = {}
        for param in common_params:
            sensitivities = []
            for parser in parsers:
                sens = parser.sensitivity.get(param, {})
                read_val = abs(sens.get('read', 0.0))
                write_val = abs(sens.get('write', 0.0))
                sensitivities.append(read_val)
                sensitivities.append(write_val)
            param_avg_sensitivity[param] = sum(sensitivities) / len(sensitivities) if sensitivities else 0.0
        
        # Select top 7 parameters with highest average sensitivity
        sorted_params = sorted(param_avg_sensitivity.items(), key=lambda x: x[1], reverse=True)
        params_to_use = [param for param, _ in sorted_params[:7]]
        
        with open(output_file, 'w') as f:
            f.write("\\begin{table*}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Normalized Sensitivity of Read/Write Latency w.r.t.\\ a few of the technology parameters. ")
            f.write("Sensitivities are normalized using $S_{\\text{norm}} = \\left(\\frac{\\partial L}{\\partial p}\\right)\\left(\\frac{p_0}{L_0}\\right)$, ")
            f.write("representing the percent change in latency per percent change in the parameter.}\n")
            f.write("\\label{tab:norm_sens_common}\n")
            f.write("\\renewcommand{\\arraystretch}{1.35}\n")
            f.write("\\scriptsize\n")
            
            # Count columns: 1 for parameter name + 2 per parser (read + write)
            num_cols = 1 + 2 * len(parsers)
            f.write("\\begin{tabularx}{\\textwidth}{l *{" + str(num_cols) + "}{>{\\centering\\arraybackslash}X}}\n")
            f.write("\\toprule\n")
            
            # Header
            f.write("\\textbf{Parameter}")
            for parser in parsers:
                f.write(f" & \\textbf{{{parser.tech_name} Read}} & \\textbf{{{parser.tech_name} Write}}")
            f.write(" \\\\\n")
            f.write("\\midrule\n\n")
            
            # Data rows
            for param in params_to_use:
                # Format parameter name for LaTeX
                param_display = param.replace('_', '\\_')
                f.write(f"{param_display}\n")
                
                for parser in parsers:
                    sens = parser.sensitivity.get(param, {})
                    read_val = sens.get('read', 0.0)
                    write_val = sens.get('write', 0.0)
                    
                    read_str = LaTeXTableGenerator.format_scientific(read_val)
                    write_str = LaTeXTableGenerator.format_scientific(write_val)
                    
                    f.write(f" & {read_str} & {write_str}")
                
                f.write(" \\\\\n\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabularx}\n")
            f.write("\\end{table*}\n")
    
    @staticmethod
    def generate_validation_table(parsers: List[WorkflowOutputParser], output_file: str):
        """Generate validation table comparing Python vs C++"""
        with open(output_file, 'w') as f:
            f.write("\\begin{table*}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Bank-level timing comparison between Python and C++ implementations for ")
            tech_names = [p.tech_name for p in parsers]
            f.write(", ".join(tech_names))
            f.write(".\n")
            f.write("Relative error is computed as $(\\text{C++} - \\text{Python}) / \\text{Python}$.}\n")
            f.write("\\label{tab:bank_timing_py_cpp}\n")
            f.write("\\renewcommand{\\arraystretch}{1.2}\n")
            f.write("\\scriptsize\n")
            
            # Table columns: Component + Unit + (Py + C++) for each tech
            num_cols = 2 + 2 * len(parsers)
            f.write("\\begin{tabularx}{\\textwidth}{l c *{" + str(num_cols) + "}{>{\\centering\\arraybackslash}X}}\n")
            f.write("\\toprule\n")
            
            # Header
            f.write("\\textbf{Component} & \\textbf{Unit}")
            for parser in parsers:
                f.write(f" & \\textbf{{{parser.tech_name} Py}} & \\textbf{{{parser.tech_name} C++}}")
            f.write(" \\\\\n")
            f.write("\\midrule\n\n")
            
            # Timing rows
            timing_components = [
                ('Total Read Latency', 'ns', 'read'),
                ('Total Write Latency', 'ns', 'write'),
                ('H-Tree / Non-H-Tree Latency', 'ps', 'htree'),
                ('Mat Latency', 'ns', 'mat'),
            ]
            
            for comp_name, unit, key in timing_components:
                f.write(f"{comp_name}   & {unit}\n")
                for parser in parsers:
                    py_val = parser.timing.get(f'py_{key}', None)
                    cpp_val = parser.timing.get(f'cpp_{key}', None)
                    
                    if unit == 'ps' and py_val is not None:
                        py_val = py_val * 1000  # Convert ns to ps
                    if unit == 'ps' and cpp_val is not None:
                        cpp_val = cpp_val * 1000  # Convert ns to ps
                    
                    py_str = f"{py_val:.3f}" if py_val is not None else "N/A"
                    cpp_str = f"{cpp_val:.3f}" if cpp_val is not None else "N/A"
                    
                    f.write(f" & {py_str} & {cpp_str}")
                f.write(" \\\\\n\n")
            
            # Sub-components (indented)
            sub_components = [
                ('Predecoder', 'ps', 'predecoder'),
                ('Subarray', 'ns', 'subarray'),
                ('Row Decoder', 'ns', 'row_decoder'),
                ('Bitline', 'ns', 'bitline'),
                ('Sense Amp', 'ps', 'senseamp'),
                ('Mux', 'ps', 'mux'),
            ]
            
            for comp_name, unit, key in sub_components:
                f.write(f"\\quad {comp_name}     & {unit}\n")
                for parser in parsers:
                    py_val = parser.timing.get(f'py_{key}', None)
                    cpp_val = parser.timing.get(f'cpp_{key}', None)
                    
                    if unit == 'ps' and py_val is not None:
                        py_val = py_val * 1000  # Convert ns to ps
                    if unit == 'ps' and cpp_val is not None:
                        cpp_val = cpp_val * 1000  # Convert ns to ps
                    
                    py_str = f"{py_val:.3f}" if py_val is not None else "N/A"
                    cpp_str = f"{cpp_val:.3f}" if cpp_val is not None else "N/A"
                    
                    f.write(f" & {py_str} & {cpp_str}")
                f.write(" \\\\\n\n")
            
            # Relative Error rows
            f.write("\\midrule\n")
            f.write("\\textbf{Relative Error (Read)}  & \\%\n")
            for parser in parsers:
                py_read = parser.timing.get('py_read')
                cpp_read = parser.timing.get('cpp_read')
                if py_read and cpp_read and py_read > 0:
                    rel_error = ((cpp_read - py_read) / py_read) * 100
                    error_str = f"${rel_error:+.2f}\\%$"
                else:
                    error_str = "--"
                f.write(f" & {error_str} & --")
            f.write(" \\\\\n\n")
            
            f.write("\\textbf{Relative Error (Write)} & \\%\n")
            for parser in parsers:
                py_write = parser.timing.get('py_write')
                cpp_write = parser.timing.get('cpp_write')
                if py_write and cpp_write and py_write > 0:
                    rel_error = ((cpp_write - py_write) / py_write) * 100
                    error_str = f"${rel_error:+.2f}\\%$"
                else:
                    error_str = "--"
                f.write(f" & {error_str} & --")
            f.write(" \\\\\n")
            
            f.write("\\bottomrule\n\n")
            f.write("\\end{tabularx}\n")
            f.write("\\end{table*}\n")
    
    @staticmethod
    def generate_optimization_table(parsers: List[WorkflowOutputParser], output_file: str):
        """Generate optimization results table showing before/after values"""
        # Collect all base and non-base parameters across all parsers
        all_base_params = set()
        all_non_base_params = set()
        for parser in parsers:
            all_base_params.update(parser.optimization.get('base_params', {}).keys())
            all_non_base_params.update(parser.optimization.get('non_base_params', {}).keys())
        
        # Only include parameters that exist in at least one parser
        base_params = sorted(all_base_params)
        non_base_params = sorted(all_non_base_params)
        
        with open(output_file, 'w') as f:
            f.write("\\begin{table*}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Optimization Results: Parameter Changes from IPOPT Optimization for ")
            tech_names = [p.tech_name for p in parsers]
            f.write(", ".join(tech_names))
            f.write(". Values show the change from initial (before) to optimized (after) values.}\n")
            f.write("\\label{tab:optimization_results}\n")
            f.write("\\renewcommand{\\arraystretch}{1.2}\n")
            f.write("\\scriptsize\n")
            
            # Count columns: 1 for parameter name + 2 per parser (before + after)
            num_cols = 1 + 2 * len(parsers)
            f.write("\\begin{tabularx}{\\textwidth}{l *{" + str(num_cols) + "}{>{\\centering\\arraybackslash}X}}\n")
            f.write("\\toprule\n")
            
            # Header
            f.write("\\textbf{Parameter}")
            for parser in parsers:
                f.write(f" & \\textbf{{{parser.tech_name} Before}} & \\textbf{{{parser.tech_name} After}}")
            f.write(" \\\\\n")
            f.write("\\midrule\n\n")
            
            # Base Parameters Section
            if base_params:
                f.write("\\multicolumn{" + str(num_cols) + "}{l}{\\textbf{Base Parameters (Tunable Knobs)}} \\\\\n")
                f.write("\\midrule\n")
                
                for param in base_params:
                    # Format parameter name for LaTeX
                    param_display = param.replace('_', '\\_')
                    f.write(f"{param_display}\n")
                    
                    for parser in parsers:
                        opt_data = parser.optimization.get('base_params', {}).get(param)
                        if opt_data:
                            before = opt_data['before']
                            after = opt_data['after']
                            # Format numbers appropriately
                            before_str = LaTeXTableGenerator._format_number(before)
                            after_str = LaTeXTableGenerator._format_number(after)
                            f.write(f" & {before_str} & {after_str}")
                        else:
                            f.write(" & -- & --")
                    
                    f.write(" \\\\\n\n")
            
            # Non-Base Parameters Section
            if non_base_params:
                f.write("\\midrule\n")
                f.write("\\multicolumn{" + str(num_cols) + "}{l}{\\textbf{Non-Base Parameters (Derived Values)}} \\\\\n")
                f.write("\\midrule\n")
                
                for param in non_base_params:
                    # Format parameter name for LaTeX
                    param_display = param.replace('_', '\\_')
                    # Get unit from first parser that has this parameter
                    unit = None
                    for parser in parsers:
                        opt_data = parser.optimization.get('non_base_params', {}).get(param)
                        if opt_data and 'unit' in opt_data:
                            unit = opt_data['unit']
                            break
                    
                    if unit:
                        param_display += f" ({unit})"
                    
                    f.write(f"{param_display}\n")
                    
                    for parser in parsers:
                        opt_data = parser.optimization.get('non_base_params', {}).get(param)
                        if opt_data:
                            before = opt_data['before']
                            after = opt_data['after']
                            # Format numbers appropriately
                            before_str = LaTeXTableGenerator._format_number(before)
                            after_str = LaTeXTableGenerator._format_number(after)
                            f.write(f" & {before_str} & {after_str}")
                        else:
                            f.write(" & -- & --")
                    
                    f.write(" \\\\\n\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabularx}\n")
            f.write("\\end{table*}\n")
    
    @staticmethod
    def _format_number(value: float) -> str:
        """Format a number for LaTeX table, using scientific notation if needed"""
        if abs(value) < 1e-3 or abs(value) > 1e6:
            # Use scientific notation
            exp = int(f"{value:.3e}".split('e')[1])
            mantissa = value / (10 ** exp)
            return f"${mantissa:.3f}\\times10^{{{exp}}}$"
        else:
            # Use regular decimal notation
            if abs(value) < 1:
                return f"${value:.6f}$"
            else:
                return f"${value:.3f}$"


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from workflow output files')
    parser.add_argument('--output-dir', type=str, default=os.path.join(os.path.dirname(__file__), 'test_output'),
                       help='Directory to write output LaTeX files')
    #parser.add_argument('--files', nargs='+', required=True,
    #                   help='Workflow output files to process')
    parser.add_argument('--table-types', nargs='+', 
                       choices=['configuration', 'sensitivity', 'validation', 'optimization', 'all'],
                       default=['all'],
                       help='Types of tables to generate')
    
    args = parser.parse_args()

    workflow_output_dir = os.path.join(os.path.dirname(__file__), '..', 'workflow_output')

    files = [
        #f'{workflow_output_dir}/sample_SRAM_2layer_out.txt',
        #f'{workflow_output_dir}/sample_SRAM_4layer_out.txt',
        #f'{workflow_output_dir}/sample_STTRAM_out.txt',
        #f'{workflow_output_dir}/sample_2D_eDRAM_out.txt',
        #f'{workflow_output_dir}/sample_2DReRAM_out.txt',
        #f'{workflow_output_dir}/sample_PCRAM_out.txt',
        #f'{workflow_output_dir}/sample_2D_eDRAM_notech_out.txt',
        #f'{workflow_output_dir}/sample_2DReRAM_notech_out.txt',
        #f'{workflow_output_dir}/sample_SRAM_2layer_notech_out.txt',
        f'{workflow_output_dir}/sample_SRAM_2layer_nowire_out.txt',
    ]

    output_identifier = ""
    for file in files:
        output_identifier += "_" + file.split('/')[-1].replace('_out.txt', '').replace('sample_', '')
    
    # Parse all files
    parsers = []
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        try:
            parsers.append(WorkflowOutputParser(file_path))
            print(f"Parsed: {file_path}")
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            continue
    
    if not parsers:
        print("Error: No files were successfully parsed")
        return
    
    # Generate tables
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    table_types = args.table_types
    if 'all' in table_types:
        table_types = ['configuration', 'sensitivity', 'validation', 'optimization']
    print(f"Generating tables for {table_types} into {output_dir}")
    for table_type in table_types:
        table_output_dir = output_dir / table_type
        if not os.path.exists(table_output_dir):
            table_output_dir.mkdir(parents=True, exist_ok=True)
        if table_type == 'configuration':
            LaTeXTableGenerator.generate_configuration_table(parsers, str(table_output_dir / f'configuration{output_identifier}.tex'))
            print(f"Generated: {table_output_dir / f'configuration{output_identifier}.tex'}")
        elif table_type == 'sensitivity':
            LaTeXTableGenerator.generate_sensitivity_table(parsers, str(table_output_dir / f'sensitivity{output_identifier}.tex'))
            print(f"Generated: {table_output_dir / f'sensitivity{output_identifier}.tex'}")
        elif table_type == 'validation':
            LaTeXTableGenerator.generate_validation_table(parsers, str(table_output_dir / f'validation{output_identifier}.tex'))
            print(f"Generated: {table_output_dir / f'validation{output_identifier}.tex'}")
        elif table_type == 'optimization':
            LaTeXTableGenerator.generate_optimization_table(parsers, str(table_output_dir / f'optimization{output_identifier}.tex'))
            print(f"Generated: {table_output_dir / f'optimization{output_identifier}.tex'}")

if __name__ == '__main__':
    main()

