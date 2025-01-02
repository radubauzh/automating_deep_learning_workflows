import sys
import os
import json
import datetime
import re
from itertools import product
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QProgressBar, QTextEdit
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QProcess

# Function to create a product of dictionaries
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))

class ConfigGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.process = None  # Initialize the process attribute
        self.total_epochs = 0  # To track the total number of epochs
        self.current_epoch = 0  # To track the current epoch
        self.total_experiments = 0  # To track total experiments
        self.current_experiment = 1  # Start from 1 to avoid division by zero

    def init_ui(self):
        # Set the window title and size
        self.setWindowTitle('Modern JSON Config Generator')
        self.setGeometry(100, 100, 500, 600)

        # Layout
        layout = QVBoxLayout()

        # Font style for the labels
        label_font = QFont("Arial", 12, QFont.Bold)

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
        self.opt_name_entry.addItems(['adam', 'sgd'])
        self.add_row(layout, opt_name_label, self.opt_name_entry)

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)
        self.progress_bar.setValue(0)

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
        self.opt_name_entry.setCurrentText(inputs.get('opt_name', 'adam'))

    # Validation Functions as Instance Methods
    def validate_positive_int(self, value):
        try:
            int_value = int(value)
            if int_value > 0:
                return int_value
            else:
                return None
        except ValueError:
            return None

    def validate_non_negative_float(self, value):
        try:
            float_value = float(value)
            if float_value >= 0:
                return float_value
            else:
                return None
        except ValueError:
            return None

    def validate_comma_separated_list(self, value, validation_function):
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
        """ Generate the JSON configuration from the input values and run the script """
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

        # Ensure that the data types match what main.py expects
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

        config_dir = os.path.join(os.path.dirname(__file__), 'Configs')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, filename)

        # Save to a file
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            QMessageBox.information(self, 'Success', f'Configuration saved successfully at {config_path}!', QMessageBox.Ok)
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

    def run_script(self, config_path):
        """ Run the commands separately to check if they work """
        main_dir = os.path.dirname(os.path.abspath(__file__))

        # command = f"""
        # cd {main_dir} && \
        # conda activate DL && \
        # python -u main.py --config "{config_path}"
        # """

        command = f"""
        cd {main_dir} && \
        source ~/miniconda3/etc/profile.d/conda.sh && \
        conda activate DL && \
        python -u main.py --config "{config_path}"
        """

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)  # Combine stdout and stderr
        self.process.finished.connect(self.process_finished)
        self.process.readyRead.connect(self.read_stdout)
        self.process.errorOccurred.connect(self.process_error)
        self.process.start('/bin/bash', ['-c', command])

    def update_progress_bar(self):
        """ Update the progress bar value based on epochs and experiments """
        if self.total_epochs > 0 and self.total_experiments > 0:
            total_epoch_progress = (self.current_epoch / self.total_epochs)
            total_experiment_progress = ((self.current_experiment - 1) / self.total_experiments)

            # Combine epoch and experiment progress
            total_progress = (total_experiment_progress + total_epoch_progress / self.total_experiments) * 100

            self.progress_bar.setValue(int(total_progress))
            self.progress_bar.setFormat(f"Experiment {self.current_experiment}/{self.total_experiments} - Epoch {self.current_epoch}/{self.total_epochs}")
            QApplication.processEvents()

    def process_finished(self):
        self.progress_bar.setValue(100)
        QMessageBox.information(self, 'Success', 'Script ran successfully!', QMessageBox.Ok)
        window.close()

    def read_stdout(self):
        """ Read and process the output to update progress based on the epoch and experiment """
        output = self.process.readAll().data().decode()
        self.output_text.append(output)

        epoch_match = re.search(r'Train Epoch: (\d+)', output)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
            self.update_progress_bar()

        experiment_match = re.search(r'Running Experiment (\d+)', output)
        if experiment_match:
            self.current_experiment = int(experiment_match.group(1))
            self.update_progress_bar()

    def process_error(self, error):
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

# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConfigGenerator()

    test_inputs = {
        'batchsize': 64,
        'lr': '0.01,0.001',
        'n_epochs': 10,
        'l2_sum_lambda': '0.1,0.2',
        'l2_mul_lambda': '0.05',
        'wn': '0.9',
        'seed': '12,34',
        'depth_normalization': 'True',
        'features_normalization': 'None',
        'batch_norm': 'True',
        'bias': 'False',
        'opt_name': 'sgd'
    }
    window.set_predefined_inputs(test_inputs)

    window.show()
    sys.exit(app.exec_())