# from PyQt5.QtWidgets import QApplication, QTableView, QVBoxLayout, QWidget
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QStandardItemModel, QStandardItem

# class TableViewExample(QWidget):
#     def __init__(self):
#         super().__init__()

#         # 设置窗口布局
#         layout = QVBoxLayout()

#         # 创建 QTableView 控件
#         table_view = QTableView()

#         # 创建标准项模型 (QStandardItemModel)
#         model = QStandardItemModel(10, 3)  # 4行3列
#         model.setHorizontalHeaderLabels(['Name', 'Age', 'City'])  # 设置列标题

#         # 向模型中添加数据
#         data = [
#             ["John Doe", "30", "New York"],
#             ["Jane Smith", "25", "Los Angeles"],
#             ["Emily Johnson", "35", "Chicago"],
#             ["Michael Brown", "40", "San Francisco"]
#         ]

#         # 将数据添加到模型中
#         for row in range(4):
#             for col in range(3):
#                 item = QStandardItem(data[row][col])  # 创建标准项
#                 item.setTextAlignment(Qt.AlignCenter)  # 设置文本居中对齐
#                 model.setItem(row, col, item)  # 设置单元格的数据

#         # 将模型设置到表格视图中
#         table_view.setModel(model)

#         # 设置视图的一些样式（如列标题的背景颜色）
#         table_view.horizontalHeader().setStyleSheet("QHeaderView::section { background-color: lightgray; font-weight: bold; }")

#         # 添加 QTableView 到窗口布局
#         layout.addWidget(table_view)

#         # 设置窗口的布局
#         self.setLayout(layout)

#         # 设置窗口标题
#         self.setWindowTitle("QTableView Example")
#         self.resize(600, 300)

# # 创建并启动应用程序
# app = QApplication([])
# window = TableViewExample()
# window.show()
# app.exec_()


from PyQt5.QtWidgets import QApplication, QTableView, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem

class TableViewExample(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口布局
        layout = QVBoxLayout()

        # 创建 QTableView 控件
        table_view = QTableView()

        # 创建标准项模型 (QStandardItemModel)
        model = QStandardItemModel(1, 6)  # 1行6列
        model.setHorizontalHeaderLabels(['Column 1', 'Column 2', 'Column 3', 'Column 4', 'Column 5', 'Column 6'])  # 设置列标题

        # 向模型中添加数据
        for col in range(6):
            item = QStandardItem(f"Item {col+1}")  # 创建标准项
            item.setTextAlignment(Qt.AlignCenter)  # 设置文本居中对齐
            model.setItem(0, col, item)  # 设置数据到第一行的每一列

        # 将模型设置到表格视图中
        table_view.setModel(model)

        # 隐藏行号（第一列的默认序号）
        table_view.verticalHeader().setVisible(False)

        # 设置视图的一些样式（如列标题的背景颜色）
        table_view.horizontalHeader().setStyleSheet("QHeaderView::section { background-color: lightgray; font-weight: bold; }")

        # 添加 QTableView 到窗口布局
        layout.addWidget(table_view)

        # 设置窗口的布局
        self.setLayout(layout)

        # 设置窗口标题
        self.setWindowTitle("QTableView Example")
        self.resize(600, 300)

# 创建并启动应用程序
app = QApplication([])
window = TableViewExample()
window.show()
app.exec_()
