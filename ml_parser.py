"""
AST-based ML workflow detector
Provides precision through syntax analysis
"""
import ast
from typing import Dict


class MLParser:
    # Frameworks to detect
    ML_FRAMEWORKS = {
        'sklearn', 'scikit-learn', 'pytorch', 'torch', 'tensorflow', 'keras'
    }
    
    # Common ML model classes
    ML_CLASSES = {
        'LogisticRegression', 'LinearRegression', 'Ridge', 'Lasso',
        'RandomForestClassifier', 'RandomForestRegressor',
        'GradientBoostingClassifier', 'GradientBoostingRegressor',
        'SVC', 'SVR', 'KNeighborsClassifier', 'DecisionTreeClassifier',
        'XGBClassifier', 'XGBRegressor', 'LGBMClassifier', 'LGBMRegressor',
        'Sequential', 'Model'
    }
    
    def __init__(self):
        self.framework = None
        self.model_names = []
        self.hyperparameters = {}
        self.known_ml_classes = set()
        self.model_variables = set()
        
        # Store line numbers for each stage
        self.data_loading_lines = []
        self.training_lines = []
        self.evaluation_lines = []
        
    def parse_file(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            tree = ast.parse(code)
        except Exception as e:
            print(f"Error parsing file: {e}")
        
        # detection steps
        self._detect_imports(tree)
        self._detect_data_loading(tree)
        self._detect_models(tree)
        self._detect_training(tree)
        self._detect_evaluation(tree)
        
        result = self._build_result()
        self._print_results(result)
        return result['is_ml_workflow']
        

    def _detect_imports(self, tree: ast.AST):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._check_ml_import(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self._check_ml_import(node.module)
                    for alias in node.names:
                        if alias.name in self.ML_CLASSES:
                            self.known_ml_classes.add(alias.name)
    

    def _check_ml_import(self, module_name: str):
        for framework in self.ML_FRAMEWORKS:
            if framework in module_name.lower():
                self.framework = framework
                break
    
    def _detect_data_loading(self, tree: ast.AST):
        data_functions = {'read_csv', 'read_excel', 'load_data', 'train_test_split'}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                if func_name in data_functions:
                    lineno = getattr(node, 'lineno', 0)
                    self.data_loading_lines.append(lineno)
    
    def _detect_models(self, tree: ast.AST):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    class_name = self._get_function_name(node.value.func)
                    
                    # Check if it's an ML class
                    if class_name in self.ML_CLASSES or class_name in self.known_ml_classes:
                        self.model_names.append(class_name)
                        
                        # Track the variable name
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.model_variables.add(target.id)
                        
                        # Extract hyperparameters
                        for kw in node.value.keywords:
                            if isinstance(kw.value, ast.Constant):
                                self.hyperparameters[kw.arg] = kw.value.value
    
    def _detect_training(self, tree: ast.AST):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    method = node.func.attr
                    
                    if method in {'fit', 'train', 'fit_transform'}:
                        caller = self._get_caller_name(node.func.value)
                        
                        if caller in self.model_variables or method == 'fit':
                            lineno = getattr(node, 'lineno', 0)
                            self.training_lines.append(lineno)
    
    def _detect_evaluation(self, tree: ast.AST):
        #Detect evaluation operations
        eval_functions = {
            'score', 'evaluate', 'predict', 
            'accuracy_score', 'precision_score', 'f1_score',
            'mean_squared_error', 'r2_score'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                if func_name in eval_functions:
                    lineno = getattr(node, 'lineno', 0)
                    self.evaluation_lines.append(lineno)
    
    def _get_function_name(self, node) -> str:
        #Extract function/method name from AST node
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ''
    
    def _get_caller_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        return ''
    
    def _build_result(self) -> Dict:
        is_ml = bool(self.framework and (self.model_names or self.training_lines))
        
        return {
            'is_ml_workflow': is_ml,
            'framework': self.framework,
            'model_names': self.model_names,
            'hyperparameters': self.hyperparameters,
            'stages': {
                'data_loading': self.data_loading_lines,
                'training': self.training_lines,
                'evaluation': self.evaluation_lines
            }
        }
    
    def _print_results(self, result: Dict):
        print(f"ML Workflow: {result['is_ml_workflow']}")
        print(f"Framework: {result['framework'] or 'Not detected'}")
        print(f"Models: {result['model_names'] or 'None found'}")
        
        if result['hyperparameters']:
            print("\nHyperparameters:")
            for k, v in result['hyperparameters'].items():
                print(f"  {k}: {v}")
        
        stages = result['stages']
        print(f"\nData Loading (lines): {stages['data_loading'] or 'None'}")
        print(f"Training (lines): {stages['training'] or 'None'}")
        print(f"Evaluation (lines): {stages['evaluation'] or 'None'}")
