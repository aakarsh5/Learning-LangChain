from tkinter import Tk, filedialog

root = Tk()
root.withdraw()  # Hide root window

file_path = filedialog.askopenfilename()
print(file_path)


#############################################

from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

app = QApplication(sys.argv)  # Needed for Qt
file_path, _ = QFileDialog.getOpenFileName(
    None,
    "Select a file",
    "",
    "All Files (*)"
)

if file_path:
    print("Selected:", file_path)
else:
    print("No file selected")
