import sys
import os
import json
import datetime
import re
from itertools import product
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QTextEdit
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QProcess

# Function to create a product of dictionaries
def product_dict(**kwargs):
    """
    Creates cartesian products for the given keyword arguments. 
    Special handling for 'l2_sum_lambda' and 'l2_mul_lambda' so 
    that they don't both get set in the same dictionary (they're treated as mutually exclusive).
    """
    keys = kwargs.keys()
    if 'l2_sum_lambda' in keys and 'l2_mul_lambda' in keys:
        l2_sum_lambda_values = kwargs['l2_sum_lambda']
        l2_mul_lambda_values = kwargs['l2_mul_lambda']
        other_kwargs = {k: v for k, v in kwargs.items() if k not in ['l2_sum_lambda', 'l2_mul_lambda']}
        
        # Yield dicts where l2_sum_lambda is set, l2_mul_lambda is empty
        for l2_sum_lambda in l2_sum_lambda_values:
            for instance in product(*other_kwargs.values()):
                yield dict(zip(other_kwargs.keys(), instance), l2_sum_lambda=l2_sum_lambda, l2_mul_lambda=[])
        
        # Yield dicts where l2_mul_lambda is set, l2_sum_lambda is empty
        for l2_mul_lambda in l2_mul_lambda_values:
            for instance in product(*other_kwargs.values()):
                yield dict(zip(other_kwargs.keys(), instance), l2_sum_lambda=[], l2_mul_lambda=l2_mul_lambda)
    else:
        # Standard cartesian product for all other keyword arguments
        for instance in product(*kwargs.values()):
            yield dict(zip(keys, instance))


class ConfigGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.process = None  
        self.total_epochs = 0 
        self.current_epoch = 0
        self.total_experiments = 0 
        self.current_experiment = 1
        self.global_epoch_count = 0
        self._last_experiment = 1
        self._last_epoch_for_experiment = 0

    def init_ui(self):
        # Set the window title and size
        self.setWindowTitle('Modern JSON Config Generator')
        self.setGeometry(100, 100, 500, 600)

        # Layout
        layout = QVBoxLayout()

        # Font style for the labels
        label_font = QFont("Arial", 12, QFont.Bold)

        # Conda Environment
        env_label = QLabel('Conda Environment Name:')
        env_label.setFont(label_font)
        self.env_entry = QLineEdit('DL')  # Default name can be changed by the user
        self.add_row(layout, env_label, self.env_entry)

        # Batch Size
        batch_label = QLabel('Batch Size:')
        batch_label.setFont(label_font)
        self.batchsize_entry = QLineEdit('128')
        self.add_row(layout, batch_label, self.batchsize_entry)

        # Learning Rates
        lr_label = QLabel('Learning Rates (comma separated):')
        lr_label.setFont(label_font)
        self.lr_entry = QLineEdit('0.01')
        self.add_row(layout, lr_label, self.lr_entry)

        # Number of Epochs
        n_epochs_label = QLabel('Number of Epochs:')
        n_epochs_label.setFont(label_font)
        self.n_epochs_entry = QLineEdit('5')
        self.add_row(layout, n_epochs_label, self.n_epochs_entry)

        # L2 Sum Lambda
        l2_sum_label = QLabel('L2 Sum Lambda (comma separated):')
        l2_sum_label.setFont(label_font)
        self.l2_sum_lambda_entry = QLineEdit('')
        self.add_row(layout, l2_sum_label, self.l2_sum_lambda_entry)

        # L2 Mul Lambda
        l2_mul_label = QLabel('L2 Mul Lambda (comma separated):')
        l2_mul_label.setFont(label_font)
        self.l2_mul_lambda_entry = QLineEdit('')
        self.add_row(layout, l2_mul_label, self.l2_mul_lambda_entry)

        # WN
        wn_label = QLabel('WN (comma separated):')
        wn_label.setFont(label_font)
        self.wn_entry = QLineEdit('0.8,1')
        self.add_row(layout, wn_label, self.wn_entry)

        # Seeds
        seeds_label = QLabel('Seeds (comma separated):')
        seeds_label.setFont(label_font)
        self.seeds_entry = QLineEdit('')
        self.add_row(layout, seeds_label, self.seeds_entry)

        # Depth Normalization
        depth_label = QLabel('Depth Normalization:')
        depth_label.setFont(label_font)
        self.depth_normalization_entry = QComboBox()
        self.depth_normalization_entry.addItems(['False', 'True'])
        self.add_row(layout, depth_label, self.depth_normalization_entry)

        # Features Normalization
        features_label = QLabel('Features Normalization:')
        features_label.setFont(label_font)
        self.features_normalization_entry = QComboBox()
        self.features_normalization_entry.addItems(['f_out', 'None'])
        self.add_row(layout, features_label, self.features_normalization_entry)

        # Batch Norm
        batch_norm_label = QLabel('Batch Norm:')
        batch_norm_label.setFont(label_font)
        self.batch_norm_entry = QComboBox()
        self.batch_norm_entry.addItems(['False', 'True'])
        self.add_row(layout, batch_norm_label, self.batch_norm_entry)

        # Bias
        bias_label = QLabel('Bias:')
        bias_label.setFont(label_font)
        self.bias_entry = QComboBox()
        self.bias_entry.addItems(['True', 'False'])
        self.add_row(layout, bias_label, self.bias_entry)

        # Optimizer Name
        opt_name_label = QLabel('Optimizer Name:')
        opt_name_label.setFont(label_font)
        self.opt_name_entry = QComboBox()
        self.opt_name_entry.addItems(['adam', 'adamw', 'sgd'])
        self.add_row(layout, opt_name_label, self.opt_name_entry)

        # Output TextEdit
        self.output_text = QTextEdit(self)
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        # Generate JSON Button
        generate_btn = QPushButton('Generate JSON and Run Script')
        generate_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; padding: 10px 20px;")
        generate_btn.clicked.connect(self.create_json)
        layout.addWidget(generate_btn)

        # Set the layout
        self.setLayout(layout)

    def add_row(self, layout, label, widget):
        """ Helper function to add a label and input field in a row """
        row_layout = QHBoxLayout()
        row_layout.addWidget(label)
        row_layout.addWidget(widget)
        layout.addLayout(row_layout)

    def set_predefined_inputs(self, inputs):
        """ Set predefined inputs for testing purposes """
        self.env_entry.setText(inputs.get('env_name', 'DL'))
        self.batchsize_entry.setText(str(inputs.get('batchsize', '128')))
        self.lr_entry.setText(inputs.get('lr', '0.01'))
        self.n_epochs_entry.setText(str(inputs.get('n_epochs', '5')))
        self.l2_sum_lambda_entry.setText(inputs.get('l2_sum_lambda', ''))
        self.l2_mul_lambda_entry.setText(inputs.get('l2_mul_lambda', ''))
        self.wn_entry.setText(inputs.get('wn', '0.8,1'))
        self.seeds_entry.setText(inputs.get('seed', ''))
        self.depth_normalization_entry.setCurrentText(inputs.get('depth_normalization', 'False'))
        self.features_normalization_entry.setCurrentText(inputs.get('features_normalization', 'f_out'))
        self.batch_norm_entry.setCurrentText(inputs.get('batch_norm', 'False'))
        self.bias_entry.setCurrentText(inputs.get('bias', 'True'))
        self.opt_name_entry.setCurrentText(inputs.get('opt_name', 'adamw'))

    # Validation Functions as Instance Methods
    def validate_positive_int(self, value):
        """
        Validate that value is a positive integer. 
        Return the integer if valid, else None.
        """
        try:
            int_value = int(value)
            if int_value > 0:
                return int_value
            else:
                return None
        except ValueError:
            return None

    def validate_non_negative_float(self, value):
        """
        Validate that value is a non-negative float (>= 0). 
        Return the float if valid, else None.
        """
        try:
            float_value = float(value)
            if float_value >= 0:
                return float_value
            else:
                return None
        except ValueError:
            return None

    def validate_comma_separated_list(self, value, validation_function):
        """
        Takes a string of comma-separated values, 
        uses the specified validation_function on each item, 
        returns a list of validated items or None if any invalid is found.
        """
        if not value.strip():
            return []  # Return empty list if input is empty
        items = [x.strip() for x in value.split(',') if x.strip()]
        validated_items = []
        for item in items:
            validated_value = validation_function(item)
            if validated_value is None:
                return None  # Invalid item found
            validated_items.append(validated_value)
        return validated_items

    def create_json(self):
        """
        Generate the JSON configuration from the input values, 
        validate them, save them to a file, and then run the script.
        """
        # Validate Batch Size
        batchsize_str = self.batchsize_entry.text()
        batchsize = self.validate_positive_int(batchsize_str)
        if batchsize is None:
            QMessageBox.critical(self, 'Invalid Input', 'Batch size must be an integer greater than 0.', QMessageBox.Ok)
            return

        # Validate Learning Rates
        lr_str = self.lr_entry.text()
        lr_list = self.validate_comma_separated_list(lr_str, self.validate_non_negative_float)
        if lr_list is None or not lr_list:
            QMessageBox.critical(self, 'Invalid Input', 'Learning rates must be positive numbers, comma separated.', QMessageBox.Ok)
            return

        # Validate Number of Epochs
        n_epochs_str = self.n_epochs_entry.text()
        n_epochs = self.validate_positive_int(n_epochs_str)
        if n_epochs is None:
            QMessageBox.critical(self, 'Invalid Input', 'Number of epochs must be an integer greater than 0.', QMessageBox.Ok)
            return

        # Validate L2 Sum Lambda
        l2_sum_lambda_str = self.l2_sum_lambda_entry.text()
        l2_sum_lambda_list = self.validate_comma_separated_list(l2_sum_lambda_str, self.validate_non_negative_float)
        if l2_sum_lambda_list is None:
            QMessageBox.critical(self, 'Invalid Input', 'L2 Sum Lambda values must be non-negative numbers, comma separated.', QMessageBox.Ok)
            return

        # Validate L2 Mul Lambda
        l2_mul_lambda_str = self.l2_mul_lambda_entry.text()
        l2_mul_lambda_list = self.validate_comma_separated_list(l2_mul_lambda_str, self.validate_non_negative_float)
        if l2_mul_lambda_list is None:
            QMessageBox.critical(self, 'Invalid Input', 'L2 Mul Lambda values must be non-negative numbers, comma separated.', QMessageBox.Ok)
            return

        # Validate WN
        wn_str = self.wn_entry.text()
        if wn_str.strip() == "":
            wn_list = []
        else:
            wn_list = self.validate_comma_separated_list(wn_str, self.validate_non_negative_float)
            if wn_list is None or not wn_list:
                QMessageBox.critical(self, 'Invalid Input', 'WN values must be positive numbers, comma separated.', QMessageBox.Ok)
                return

        # Validate Seeds
        seeds_str = self.seeds_entry.text()
        if seeds_str.strip():
            seeds_list = self.validate_comma_separated_list(seeds_str, self.validate_positive_int)
            if seeds_list is None:
                QMessageBox.critical(self, 'Invalid Input', 'Seeds must be valid integers, comma separated (positive).', QMessageBox.Ok)
                return
        else:
            seeds_list = []

        # Collect final configuration
        config = {
            "batchsize": batchsize,
            "lr": lr_list,
            "n_epochs": n_epochs,
            "l2_sum_lambda": l2_sum_lambda_list,
            "l2_mul_lambda": l2_mul_lambda_list,
            "wn": wn_list,
            "seed": seeds_list,
            "depth_normalization": [self.depth_normalization_entry.currentText()],
            "features_normalization": [self.features_normalization_entry.currentText()],
            "batch_norm": [self.batch_norm_entry.currentText()],
            "bias": [self.bias_entry.currentText()],
            "device": "auto",
            "opt_name": self.opt_name_entry.currentText()
        }

        # Set total epochs from the config
        self.total_epochs = config["n_epochs"]

        # Generate a unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"config_{timestamp}.json"

        # Create a 'Configs' folder next to this file and save the config there
        config_dir = os.path.join(os.path.dirname(__file__), 'Configs')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, filename)

        # Save to a file
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            QMessageBox.information(self, 'Success', f'Configuration saved successfully at: \n \n {config_path}!', QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save configuration file:\n{str(e)}', QMessageBox.Ok)
            return

        # Prepare parameter lists for combinations
        l2_sum_lambda_values = config["l2_sum_lambda"] if config["l2_sum_lambda"] else [None]
        l2_mul_lambda_values = config["l2_mul_lambda"] if config["l2_mul_lambda"] else [None]

        param_combinations = list(product_dict(
            batchsize=[config["batchsize"]],
            lr=config["lr"],
            l2_sum_lambda=l2_sum_lambda_values,
            l2_mul_lambda=l2_mul_lambda_values,
            wn=config["wn"],
            depth_normalization=config["depth_normalization"],
            features_normalization=config["features_normalization"],
            batch_norm=config["batch_norm"],
            bias=config["bias"],
            opt_name=[config["opt_name"]],
        ))
        self.total_experiments = len(param_combinations)

        # Run the main.py script with the generated configuration file
        self.run_script(config_path)

    # def run_script(self, config_path):
    #     """Run the commands using the specified conda environment."""
    #     main_dir = os.path.dirname(os.path.abspath(__file__))
    #     env_name = self.env_entry.text().strip()  # Get the user-specified environment name

    #     # Check if the conda.sh file exists
    #     conda_path = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
    #     if not os.path.exists(conda_path):
    #         QMessageBox.critical(self, 'Missing File', f'conda.sh file not found at: {conda_path}', QMessageBox.Ok)
    #         return

    #     # Check if the Conda environment exists
    #     check_env_command = f"source {conda_path} && conda env list"
    #     process = QProcess(self)
    #     process.start('/bin/bash', ['-c', check_env_command])
    #     process.waitForFinished()

    #     output = process.readAll().data().decode()
    #     if env_name not in output:
    #         QMessageBox.critical(
    #             self,
    #             'Environment Not Found',
    #             f'The Conda environment "{env_name}" does not exist.\n'
    #             'Please create it using "conda create -n <env_name> python=3.x" and try again.',
    #             QMessageBox.Ok,
    #         )
    #         return

    #     # Prepare the bash command
    #     command = f"""
    #     cd {main_dir} && \
    #     source {conda_path} && \
    #     conda activate {env_name} && \
    #     python -u main.py --config "{config_path}"
    #     """

    #     self.process = QProcess(self)
    #     self.process.setProcessChannelMode(QProcess.MergedChannels)  # Combine stdout and stderr

    #     # Connect signals for process events
    #     self.process.finished.connect(self.on_process_finished)
    #     self.process.readyRead.connect(self.read_stdout)
    #     self.process.errorOccurred.connect(self.process_error)

    #     # Start the process
    #     self.process.start('/bin/bash', ['-c', command])


    def run_script(self, config_path):
        """Run the commands using the specified conda environment."""
        main_dir = os.path.dirname(os.path.abspath(__file__))
        env_name = self.env_entry.text().strip()  # Get the user-specified environment name

        # Check if the conda.sh file exists
        conda_path = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
        if not os.path.exists(conda_path):
            QMessageBox.critical(self, 'Missing File', f'conda.sh file not found at: {conda_path}', QMessageBox.Ok)
            return

        # Check if the Conda environment exists
        check_env_command = f"source {conda_path} && conda env list"
        process = QProcess(self)
        process.start('/bin/bash', ['-c', check_env_command])
        process.waitForFinished()

        output = process.readAll().data().decode()
        if env_name not in output:
            QMessageBox.critical(
                self,
                'Environment Not Found',
                f'The Conda environment "{env_name}" does not exist.\n'
                'Please create it using "conda create -n <env_name> python=3.x" and try again.',
                QMessageBox.Ok,
            )
            return

        # Check if the OpenAI API key is set in the environment
        if not os.getenv("OPENAI_API_KEY"):
            QMessageBox.critical(
                self,
                'Missing API Key',
                'The OpenAI API key is missing from the .env file.\n'
                'Please add "OPENAI_API_KEY=your_openai_api_key_here" to the .env file and try again.',
                QMessageBox.Ok,
            )
            return

        # Prepare the bash command
        command = f"""
        cd {main_dir} && \
        source {conda_path} && \
        conda activate {env_name} && \
        python -u main.py --config "{config_path}"
        """

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)  # Combine stdout and stderr

        # Connect signals for process events
        self.process.finished.connect(self.on_process_finished)
        self.process.readyRead.connect(self.read_stdout)
        self.process.errorOccurred.connect(self.process_error)

        # Start the process
        self.process.start('/bin/bash', ['-c', command])





    def on_process_finished(self):
        """
        Called when the QProcess finishes. We'll check its exit code
        to decide whether it was truly successful or not.
        """
        exit_code = self.process.exitCode()
        if exit_code == 0:
            QMessageBox.information(self, 'Success', 'Script ran successfully!', QMessageBox.Ok)
        else:
            QMessageBox.critical(
                self, 
                'Script Error', 
                f'Script did not run successfully. Exit code: {exit_code}', 
                QMessageBox.Ok
            )
        # Close the window after finishing
        self.close()

    def read_stdout(self):
        output = self.process.readAll().data().decode()
        self.output_text.append(output)

        if "ERROR:" in output:
            error_message = output.split("ERROR:", 1)[1].strip()
            QMessageBox.critical(self, 'Script Error', f'The script encountered an error:\n{error_message}', QMessageBox.Ok)

        # Check if we moved to a new experiment
        experiment_match = re.search(r'Running Experiment (\d+)', output)
        if experiment_match:
            new_experiment_num = int(experiment_match.group(1))
            if new_experiment_num != self._last_experiment:
                self._last_experiment = new_experiment_num
                self._last_epoch_for_experiment = 0  # reset for new experiment
            self.current_experiment = new_experiment_num

        # Only count new epochs if the epoch number is higher than the last recorded
        epoch_matches = re.findall(r'Train Epoch: (\d+)', output)
        for epoch_str in epoch_matches:
            epoch_num = int(epoch_str)
            if epoch_num > self._last_epoch_for_experiment:
                self._last_epoch_for_experiment = epoch_num
                self.global_epoch_count += 1

    def process_error(self, error):
        """
        If the QProcess fails to start, crashes, or times out, handle it here.
        """
        error_str = {
            QProcess.FailedToStart: "Failed to start",
            QProcess.Crashed: "Crashed",
            QProcess.Timedout: "Timed out",
            QProcess.WriteError: "Write error",
            QProcess.ReadError: "Read error",
            QProcess.UnknownError: "Unknown error"
        }.get(error, "Unknown error")

        QMessageBox.critical(self, 'Process Error', f'Process error occurred: {error_str}', QMessageBox.Ok)
        self.output_text.append(f"Process error occurred: {error_str}")
        print(f"GUI process error: {error}")



# Run the application if this file is executed directly
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConfigGenerator()

    # Example of setting some predefined inputs programmatically
    test_inputs = {
        'env_name': 'dl_workflow_env', 
        'batchsize': 64,
        'lr': '0.01',
        'n_epochs': 10,
        'l2_sum_lambda': '0.01,0.001',
        'l2_mul_lambda': '0.01,0.001',
        'wn': '0.9',
        'seed': '42',
        'depth_normalization': 'False',
        'features_normalization': 'f_out',
        'batch_norm': 'False',
        'bias': 'False',
        'opt_name': 'adamw'
    }
    window.set_predefined_inputs(test_inputs)

    window.show()
    sys.exit(app.exec_())
