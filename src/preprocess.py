import logging
import time
import os
logger = logging.getLogger(__name__)

import copy
import sympy as sp
import pyomo.environ as pyo
import pyomo.core.expr.sympy_tools as sympy_tools
from pyomo.opt import SolverFactory

from MyPyomoSympyBimap import MyPyomoSympyBimap

def symbolic_max(a, b, evaluate=True):
    """
    Max(a, b) in a format which ipopt accepts.
    """
    return 0.5 * (a + b + sp.Abs(a - b, evaluate=evaluate))

class Preprocessor:
    """
    Prepares and processes symbolic and Pyomo-based optimization models. Handles
    mapping between symbolic and Pyomo variables, applies constraints, and manages substitutions for
    technology parameters.
    """
    def __init__(self, initial_values, out_file, solver_name="ipopt"):
        """
        Initialize the Preprocessor instance, setting up mappings, initial values, and constraint sets.
        """
        self.mapping = {}
        self.pyomo_obj_exp = None
        self.initial_val = 0
        self.free_symbols = []
        self.multistart = False
        self.obj = 0
        self.initial_values = initial_values
        self.regularization = 0
        self.out_file = out_file
        self.solver_name = solver_name
    
    def pyomo_constraint(self, model, i):
        #print(f"constraint: {self.constraints[i]}")
        pyo_expr = sympy_tools.sympy2pyomo_expression(self.constraints[i], self.bimap)
        return pyo_expr

    def add_constraints(self, model):
        """
        Add all relevant constraints (obj, power, log, V_dd, etc.) to the Pyomo model.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model instance.

        Returns:
            None
        """
        logger.info("Adding Constraints")
        print(f"adding constraints. initial val: {self.initial_val};") # obj_exp: {self.pyomo_obj_exp}")
        model.Constraints = pyo.Constraint([i for i in range(len(self.constraints))], rule=self.pyomo_constraint)
        #model.Regularization = pyo.Constraint(expr=self.regularization <= self.initial_val / 50)
        return model

    def add_regularization_to_objective(self, obj):
        """
        Parameters:
        obj: sympy objective function
        """
        #l = self.initial_val / 100
        l = self.initial_val / (100 + len(self.free_symbols))
        logger.info("Adding regularization.")
        self.regularization = 0
        # normal regularization for each variable
        for symbol in self.free_symbols:
            if self.initial_values[symbol] != 0:
                self.regularization += symbolic_max((self.initial_values[symbol]/ symbol- 1), 
                                                            (symbol/self.initial_values[symbol] - 1)) ** 2

        obj += l * self.regularization
        return obj
        
    def get_solver(self, solver_name):
        if self.multistart:
            opt = SolverFactory("multistart")
            # Configure multistart solver options using the solve() method parameters
            # These will be passed when solve() is called
            self.multistart_options = {
                "solver": solver_name,
                "iterations": 10,  # Number of multistart iterations
                "strategy": "rand_guess_and_bound",  # Restart strategy: rand, midpoint_guess_and_bound, etc.
                "stopping_mass": 0.5,  # For high confidence stopping rule
                "stopping_delta": 0.5,  # For high confidence stopping rule
                "suppress_unbounded_warning": False,
                "HCS_max_iterations": 1000,  # Max iterations for high confidence stopping
                "HCS_tolerance": 0,  # Tolerance for HCS objective value equality
                "solver_args": {
                    "options": {
                        "print_level": 5, 
                        "print_info_string": "yes",
                        "output_file": self.out_file,
                        "wantsol": 2,
                        "max_iter": 500
                    }
                }
            }
        elif solver_name == "ipopt":
            opt = SolverFactory("ipopt")
            opt.options["warm_start_init_point"] = "yes"
            #opt.options['warm_start_bound_push'] = 1e-9
            #opt.options['warm_start_mult_bound_push'] = 1e-9
           # opt.options['warm_start_bound_frac'] = 1e-9
            #opt.options['warm_start_slack_bound_push'] = 1e-9
            #opt.options['warm_start_slack_bound_frac'] = 1e-9
            #opt.options['mu_init'] = 0.1
            # opt.options['acceptable_obj_change_tol'] = self.initial_val / 100
            #opt.options['tol'] = 1
            # opt.options['print_level'] = 12
            # opt.options['nlp_scaling_method'] = 'none'
            #opt.options["bound_relax_factor"] = 0
            opt.options["max_iter"] = 1500
            opt.options["print_info_string"] = "yes"
            opt.options["output_file"] = self.out_file
            opt.options["wantsol"] = 2
            opt.options["halt_on_ampl_error"] = "yes"
        elif solver_name == "trustregion":
            opt = SolverFactory("trustregion")
        else:
            raise ValueError(f"Solver {solver_name} not supported")
        print(f"output file: {self.out_file}")
        return opt

    def create_scaling(self, model):
        logger.info("Creating scaling")
        model.scaling_factor[model.obj] = 1/self.initial_val
        print(f"mapping: {self.mapping}")
        for s in self.free_symbols:
            if s in self.initial_values and self.initial_values[s] != 0:
                print(f"symbol name: {s.name}: scaling factor: {1 / self.initial_values[s]}")
                model.scaling_factor[model.x[self.mapping[s]]] = (
                    1 / self.initial_values[s]
                )
        for i in range(len(model.Constraints)):
            scaling_factor = 1/self.constraint_initial_values[i] if self.constraint_initial_values[i] != 0 else 1   
            model.scaling_factor[model.Constraints[i]] = scaling_factor
            print(f"constraint {i}: scaling factor: {scaling_factor}")

    def begin(self, model, obj, multistart, constraints):
        self.multistart = multistart
        self.free_symbols = list(obj.free_symbols)
        for i in range(len(constraints)):
            print(f"constraint {i}: {constraints[i]}")
            self.free_symbols.extend(constraints[i].free_symbols)
        self.free_symbols = list(set(self.free_symbols))

        self.constraints = constraints

        self.constraint_initial_values = [float(max(abs(constraint.lhs.xreplace(self.initial_values)), abs(constraint.rhs.xreplace(self.initial_values)))) for constraint in self.constraints]
        print(f"constraint initial values: {self.constraint_initial_values}")
        self.initial_val = float(obj.xreplace(self.initial_values))
        print(f"obj: {obj}")
        print(f"initial val: {self.initial_val}")

        print(f"length of free symbols: {len(self.free_symbols)}")

        model.nVars = pyo.Param(initialize=len(self.free_symbols))
        model.N = pyo.RangeSet(model.nVars)
        model.x = pyo.Var(model.N, domain=pyo.NonNegativeReals)
        self.mapping = {}

        i = 0
        for j in model.x:
            self.mapping[self.free_symbols[i]] = j
            print(f"x[{j}] {self.free_symbols[i]}")
            i += 1

        print("building bimap")
        m = MyPyomoSympyBimap()
        self.bimap = m
        for symbol in self.free_symbols:
            # create self.mapping of sympy symbols to pyomo symbols
            m.sympy2pyomo[symbol] = model.x[self.mapping[symbol]]
            # give pyomo symbols an inital value for warm start
            if symbol in self.initial_values:# and not symbol.name.startswith("node_arrivals_"):
                model.x[self.mapping[symbol]] = self.initial_values[symbol]
                print(f"symbol: {symbol}; initial value: {self.initial_values[symbol]}")

        print(f"converting to pyomo exp")
        start_time = time.time()
        self.pyomo_obj_exp = sympy_tools.sympy2pyomo_expression(obj, m)

        sympy_obj = self.add_regularization_to_objective(obj)
        self.regularization = sympy_tools.sympy2pyomo_expression(self.regularization, m)
        print(f"added regularization")
        print(f"value of objective after regularization: {sympy_obj.xreplace(self.initial_values)}")

        self.obj = sympy_tools.sympy2pyomo_expression(sympy_obj, m)

        logger.info(f"time to convert all exprs to pyomo: {time.time()-start_time}")
        start_time = time.time()


        model.obj = pyo.Objective(expr=self.obj, sense=pyo.minimize)

        model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        self.add_constraints(model)
        self.create_scaling(model)

        logger.info(f"time to add constraints and create scaling: {time.time()-start_time}")

        scaled_model = pyo.TransformationFactory("core.scale_model").create_using(model)

        scaled_preproc_model = pyo.TransformationFactory(
            "contrib.constraints_to_var_bounds"
        ).create_using(scaled_model)
        preproc_model = pyo.TransformationFactory(
            "contrib.constraints_to_var_bounds"
        ).create_using(model)
        opt = self.get_solver(self.solver_name)
        # Return both the solver and the options for multistart
        if self.multistart:
            return opt, scaled_preproc_model, preproc_model, getattr(self, 'multistart_options', {})
        else:
            return opt, scaled_preproc_model, preproc_model, {}    
