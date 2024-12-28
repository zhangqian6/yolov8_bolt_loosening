import sys
import cv2
import hydra
import torch
import os
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
import window 
import shutil

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode

class main0(QMainWindow,window.Ui_MainWindow):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.det = None
        self.sum = 0 # 单张图片中检测到目标的数量
        self.count = 0
        self.flag = 0 # tableview点击控制
        self.detecting = False # 控制检测状态
        self.executing = False
        self.pasued = True

        self.setupUi(self)
        self.init_table()
        self.init_fuc()

        self.doubleSpinBox.setRange(0.0, 1.0) 
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setValue(self.cfg.conf)
        self.doubleSpinBox_2.setRange(0.0, 1.0)
        self.doubleSpinBox_2.setSingleStep(0.01)
        self.doubleSpinBox_2.setValue(self.cfg.iou)

        # 存储每张图片的检测数据
        self.image_detections = {}
        self.image_path = {}
        self.image_boBox = {}
        self.image_sum = {}
        self.image_tm = {}
        self.image_allLabel = {}
        
        self.cap = None  # 视频捕获对象

        # 创建进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.hide() # 初始隐藏进度条

    def init_table(self):
        self.tb_model = QStandardItemModel() #1行6列
        self.tb_model.setHorizontalHeaderLabels(['序号','文件路径','目标编号','类别','置信度','坐标位置']) 
        self.tableView.setModel(self.tb_model)
        # 设置每列的宽度
        self.tableView.setColumnWidth(0, 150)  # 第一列宽度 100
        self.tableView.setColumnWidth(1, 250)  # 第二列宽度 150
        self.tableView.setColumnWidth(2, 150)  # 第三列宽度 200
        self.tableView.setColumnWidth(3, 150)  # 第四列宽度 250
        self.tableView.setColumnWidth(4, 150)  # 第五列宽度 300
        self.tableView.setColumnWidth(5, 250)  # 第六列宽度 350
        # 设置视图的一些样式（如列标题的背景颜色）
        self.tableView.horizontalHeader().setStyleSheet("QHeaderView::section { background-color: lightgray; font-weight: bold; }")

        # 隐藏行号（第一列的默认序号）
        self.tableView.verticalHeader().setVisible(False)
       
    def init_fuc(self):
        self.checkBox.stateChanged.connect(self.on_checkbox_state_changed)
        self.doubleSpinBox.valueChanged.connect(self.on_value_changed)
        self.doubleSpinBox_2.valueChanged.connect(self.on_value_changed2)
        self.comboBox.currentIndexChanged.connect(self.on_combobox_change)
        self.tableView.clicked.connect(self.on_item_clicked)
        self.pushButton.clicked.connect(self.open_pic)
        self.pushButton_2.clicked.connect(self.open_file)
        self.pushButton_3.clicked.connect(self.open_vedio)
        self.pushButton_4.clicked.connect(self.open_camera)
        self.pushButton_5.clicked.connect(self.save_detect_frame)
        self.pushButton_6.clicked.connect(self.exit_app)


    def on_checkbox_state_changed(self, state):
        if state == 2: # 选中状态
            self.cfg.hide_labels = False
            self.cfg.hide_conf = False
        else:
            self.cfg.hide_labels = True
            self.cfg.hide_conf = True

    def on_value_changed(self):
        # 当QDoubleSpinBox的值变化时更新标签
        self.cfg.conf = self.doubleSpinBox.value()

    def on_value_changed2(self):
        self.cfg.iou = self.doubleSpinBox_2.value()

    def on_combobox_change(self):
        selected_index = self.comboBox.currentIndex()
        if selected_index != 0:
            self.label_2.clear()
            img = cv2.imread(self.fileName1)
            annotator_one_img = self.write_one_results(selected_index, self.det, img)

            # 将numpy图片转成QImage
            height, width, channels = annotator_one_img.shape
            bytes_per_line = 3 * width
            q_image = QImage(annotator_one_img.data, width, height, bytes_per_line, QImage.Format_BGR888)
            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(q_image)
            
            self.label_2.setPixmap(pixmap.scaled(self.label_2.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            self.label_2.setText('')  # Clear the label text
        else:
            pp = self.image_allLabel[self.fileName1]
            img =cv2.imread(pp)
            # 将numpy图片转成QImage
            height, width, channels = img.shape
            bytes_per_line = 3 * width
            q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_BGR888)
            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(q_image)
            
            self.label_2.setPixmap(pixmap.scaled(self.label_2.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            self.label_2.setText('')  # Clear the label text


    def on_item_clicked(self):
        selected_index = self.tableView.selectionModel().currentIndex()
        if selected_index.isValid():
            row = selected_index.row()
            # 获取当前行数据
            row_data = []
            for col in range(self.tb_model.columnCount()):
                item = self.tb_model.item(row,col)
                row_data.append(item.text()) # 获取每一列的数据

            # 读取原图
            crop_path = row_data[1] # 获取图片路径
            image_path_obj = Path(crop_path)

            ggparent_dir = image_path_obj.parent.parent.parent
            if self.flag == 0:
                img_name = self.get_unique_images(ggparent_dir)# 获取唯一图片路径
                img_name = img_name[0]

                # img_path0是点击的原图所在地址
                img_path0 = self.image_path[img_name]
            else:
                img_path0 = self.image_path[os.path.basename(ggparent_dir)]

            self.fileName1 = img_path0

            # 修改boBox
            self.comboBox.clear()
            boBox = self.image_boBox[self.fileName1]
            self.comboBox.addItem('全部')
            self.comboBox.addItems(boBox)

            sel_index = int(row_data[2]) # 获取目标编号

            # 获取图像的det
            self.det = self.image_detections[img_path0]

            # 修改检测个数和时间
            num = self.image_sum[self.fileName1]
            tm = self.image_tm[self.fileName1]
            self.label_6.setText(str(num)) # 总目标数
            self.label_8.setText(tm + 'ms') # 用时


            # 向label_2中插入图片
            self.label_2.clear()
            img = cv2.imread(self.fileName1)
            self.comboBox.setCurrentIndex(sel_index)
            annotator_one_img = self.write_one_results(sel_index, self.det, img)
            # 将numpy图片转成QImage
            height, width, channels = annotator_one_img.shape
            bytes_per_line = 3 * width
            q_image = QImage(annotator_one_img.data, width, height, bytes_per_line, QImage.Format_BGR888)
            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(q_image)
            
            self.label_2.setPixmap(pixmap.scaled(self.label_2.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            self.label_2.setText('')  # Clear the label text


    def get_unique_images(self,folder_path):
        # 支持的图片文件格式
        valid_image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        
        # 存储所有的图片文件路径
        image_files = []
        
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # 检查文件扩展名是否为图片格式
            if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in valid_image_extensions):
                image_files.append(filename)
        
        # 返回所有的图片文件路径
        return image_files

    # 图片检测
    def open_pic(self):
        self.sum = 0
        self.flag = 0
        options = QFileDialog.Options()
        self.fileName1, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)', options=options)
        if self.fileName1:
            for model, tm, im0s in self.predict():
                self.model = model
                det = self.predictor.output['det']
                self.det = det # 显示comboBox信息

                box = det[:,:4] # 边界框
                confidences = det[:,4] # 置信度
                class_indices = det[:,5].astype(int) # 类序号
                predicted_classes = [self.model.names[idx] for idx in class_indices] # 序号转成类别名

                # 清空表格数据
                row_count = self.tb_model.rowCount() 
                self.tb_model.removeRows(0, row_count) 

                # 存储结果,向tableview和comboBox中添加信息
                class_count = {}
                new_predicted_classes = []
                self.image_detections[self.fileName1] = det 
                name = os.path.basename(self.fileName1)
                self.image_allLabel[self.fileName1] = os.path.join(self.save_dir, name)
                self.image_tm[self.fileName1] = tm
                self.image_path[name] = self.fileName1
                for box, confidences, class_name in zip(box,confidences, predicted_classes): # 修改重复的类别名
                    # 给相同类别的标签添加数字后缀
                    if class_name not in class_count:
                        class_count[class_name] = 0
                    class_count[class_name] += 1
                    
                    #创建新标签
                    new_class_name = f"{class_name}{class_count[class_name] - 1}" if class_count[class_name] > 1 else class_name
                    new_predicted_classes.append(new_class_name)
                    pp = '(' + ', '.join([str(item) for item in box]) + ')'
                    self.sum += 1
                    # 向TableView添加信息 
                    row = []
                    file=f'{self.save_dir}/crops/{new_class_name}/{new_class_name}.jpg'
                    row.append(QStandardItem(str(self.sum)))
                    row.append(QStandardItem(str(file)))
                    row.append(QStandardItem(str(self.sum)))
                    row.append(QStandardItem(str(new_class_name)))
                    row.append(QStandardItem(str(f'{confidences:.2f}')))
                    row.append(QStandardItem(str(pp)))
                    self.tb_model.appendRow(row)
                self.comboBox.clear()
                self.comboBox.addItem('全部')
                self.comboBox.addItems(new_predicted_classes)
                self.image_boBox[self.fileName1] = new_predicted_classes
                self.image_sum[self.fileName1] = self.sum
                self.image = self.predictor.annotator.result() # 标记所有的结果图
                self.label_6.setText(str(self.sum)) # 总目标数
                self.label_8.setText(tm +'ms') # 用时

                
                # show image
                height, width, channels = self.image.shape
                bytes_per_line = 3 * width
                q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(q_image)
                self.label_2.setPixmap(pixmap.scaled(self.label_2.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
                self.label_2.setText('')  # Clear the label text
            
    # 文件中图片检测
    def open_file(self):   
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder', '')
        # 清空表格数据
        row_count = self.tb_model.rowCount()
        self.tb_model.removeRows(0, row_count)
        self.flag == 0
        for filename in os.listdir(folder_path):
            self.sum = 0
            # 构造完整的文件路径
            file_path = os.path.join(folder_path, filename)

            # # 检查文件是否是图片文件，可以根据文件扩展名来判断
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.fileName1 = file_path
                try:
                    for model, tm, im in self.predict():
                        print(self.save_dir)
                        self.model = model
                        det = self.predictor.output['det']
                        self.det = det

                        # 存储结果,向tableview和comboBox中添加信息
                        self.image_detections[self.fileName1] = det # 存储检测结果
                        self.image_path[filename] = file_path
                        self.image_tm[self.fileName1] = tm
                        self.image_allLabel[self.fileName1] = os.path.join(self.save_dir, filename)
                        
                        box = det[:,:4] # 边界框
                        confidences = det[:,4] # 置信度
                        class_indices = det[:,5].astype(int) # 类序号
                        predicted_classes = [model.names[idx] for idx in class_indices] # 序号转成类别名

                        class_count = {}
                        new_predicted_classes = []
                        
                        for box, confidences, class_name in zip(box,confidences, predicted_classes): # 修改重复的类别名
                            # 给相同类别的标签添加数字后缀
                            if class_name not in class_count:
                                class_count[class_name] = 0
                            class_count[class_name] += 1

                            #创建新标签
                            new_class_name = f"{class_name}{class_count[class_name] - 1}" if class_count[class_name] > 1 else class_name
                            new_predicted_classes.append(new_class_name)
                            pp = '(' + ', '.join([str(item) for item in box]) + ')'
                            self.sum += 1
                            # 向TableView添加信息
                            row = []
                            file=f'{self.save_dir}/crops/{new_class_name}/{new_class_name}.jpg'
                            row.append(QStandardItem(str(self.sum)))
                            row.append(QStandardItem(str(file)))
                            row.append(QStandardItem(str(self.sum)))
                            row.append(QStandardItem(str(new_class_name)))
                            row.append(QStandardItem(str(f'{confidences:.2f}')))
                            row.append(QStandardItem(str(pp)))
                            self.tb_model.appendRow(row)
                        self.comboBox.clear()
                        self.comboBox.addItem('全部')
                        self.comboBox.addItems(new_predicted_classes)
                        self.image_boBox[self.fileName1] = new_predicted_classes
                        self.image_sum[self.fileName1] = self.sum
                        self.image = self.predictor.annotator.result() # 标记所有的结果图
                    
                        self.label_6.setText(str(self.sum)) # 总目标数
                        self.label_8.setText(str(round(tm*1000, 3))+'ms') # 用时


                        # 将numpy图片转成QImage
                        height, width, channels = self.image.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_BGR888)
                        # Convert QImage to QPixmap
                        pixmap = QPixmap.fromImage(q_image)
                        self.label_2.setPixmap(pixmap.scaled(self.label_2.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
                        self.label_2.setText('')  # Clear the label text
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    # 视频检测
    def open_vedio(self):
        row_count = self.tb_model.rowCount() # 清空表格数据
        self.tb_model.removeRows(0, row_count) 
        self.flag = 1
        # 打开文件对话框
        self.fileName1, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if self.fileName1:
            self.cfg.source = self.fileName1
            self.vi = False
            # 打开视频文件
            self.cap = cv2.VideoCapture(self.fileName1)
            if not self.cap.isOpened():
                print("Error: Cannot open video.")
                return
            frame_number = 1 # 帧数
            dirname = os.path.dirname(self.fileName1) # 父文件路径 E:/z2024/studying/yolov8/ultralytics-8.0.6/ultralytics/assets
            for model, tm, im0s in self.predict():
                self.sum = 0
                frame_filename = f"frame_{frame_number:04d}.jpg" # frame_0001.jpg
                frame_number += 1
                file_path = os.path.join(dirname, frame_filename)    # 每一帧原图片存在路径 E:/z2024/studying/yolov8/ultralytics-8.0.6/ultralytics/assets\frame_0092.jpg
                lab_img_path = os.path.join(self.save_dir, frame_filename) # 带所有检测结果的每一帧地址
                image = self.predictor.annotator.result() # 标记所有的结果图
                cv2.imwrite(file_path, im0s) # 保存每一帧图片
                cv2.imwrite(lab_img_path, image) # 保存结果图
                self.model = model 
                det = self.predictor.output['det']
                self.det = det

                # 存储结果,向tableview和comboBox中添加信息
                self.image_detections[file_path] = det # 存储检测结果
                self.image_path[frame_filename] = file_path # 原图存在路径

                self.image_tm[file_path] = tm
                self.image_allLabel[file_path] = lab_img_path # 检测结果图
                

                box = det[:,:4] # 边界框
                confidences = det[:,4] # 置信度
                class_indices = det[:,5].astype(int) # 类序号
                predicted_classes = [model.names[idx] for idx in class_indices] # 序号转成类别名

                class_count = {}
                new_predicted_classes = []

                for box, confidences, class_name in zip(box,confidences, predicted_classes): # 修改重复的类别名
                    # 给相同类别的标签添加数字后缀
                    if class_name not in class_count:
                        class_count[class_name] = 0
                    class_count[class_name] += 1

                    #创建新标签
                    new_class_name = f"{class_name}{class_count[class_name] - 1}" if class_count[class_name] > 1 else class_name
                    new_predicted_classes.append(new_class_name)
                    pp = '(' + ', '.join([str(item) for item in box]) + ')'
                    self.sum += 1
                    self.count += 1
                    # 向TableView添加信息 
                    row = []
                    file=f'{self.save_dir}/{frame_filename}/crops/{new_class_name}/{new_class_name}.jpg'
                    row.append(QStandardItem(str(self.count)))
                    row.append(QStandardItem(str(file)))
                    row.append(QStandardItem(str(self.sum)))
                    row.append(QStandardItem(str(new_class_name)))
                    row.append(QStandardItem(str(f'{confidences:.2f}')))
                    row.append(QStandardItem(str(pp)))
                    self.tb_model.appendRow(row)
                self.fileName1 = file_path
                self.comboBox.clear()
                self.comboBox.addItem('全部')
                self.comboBox.addItems(new_predicted_classes)
                self.image_boBox[file_path] = new_predicted_classes
                self.image_sum[file_path] = self.sum

                self.label_6.setText(str(self.sum)) # 总目标数
                # self.label_8.setText(str(round(tm*1000, 3))+'ms') # 用时
                self.label_8.setText(tm + 'ms')
                QApplication.processEvents()  # 让事件循环及时处理UI更新

    # 摄像头检测            
    def open_camera(self):
        if self.executing == False:
            self.executing = True
            row_count = self.tb_model.rowCount() # 清空表格数据
            self.tb_model.removeRows(0, row_count)
            self.flag = 1

            self.fileName1 = 0
            frame_number = 1 # 帧数
            dirname = r'E:\z2024\studying\yolov8\ultralytics-8.0.6\ultralytics\assets'
            for model, tm, im0s in self.predict():
                self.sum = 0
                frame_filename = f"frame_{frame_number:04d}.jpg" # frame_0001.jpg
                frame_number += 1
                file_path = os.path.join(dirname, frame_filename)    # 每一帧原图片存在路径 E:/z2024/studying/yolov8/ultralytics-8.0.6/ultralytics/assets\frame_0092.jpg
                lab_img_path = os.path.join(self.save_dir, frame_filename) # 带所有检测结果的每一帧地址
                image = self.predictor.annotator.result() # 标记所有的结果图
                cv2.imwrite(file_path, im0s) # 保存每一帧图片
                cv2.imwrite(lab_img_path, image) # 保存结果图
                self.model = model 
                det = self.predictor.output['det']
                self.det = det

                # 存储结果,向tableview和comboBox中添加信息
                self.image_detections[file_path] = det # 存储检测结果
                self.image_path[frame_filename] = file_path # 原图存在路径

                self.image_tm[file_path] = tm
                self.image_allLabel[file_path] = lab_img_path # 检测结果图
                

                box = det[:,:4] # 边界框
                confidences = det[:,4] # 置信度
                class_indices = det[:,5].astype(int) # 类序号
                predicted_classes = [model.names[idx] for idx in class_indices] # 序号转成类别名

                class_count = {}
                new_predicted_classes = []

                for box, confidences, class_name in zip(box,confidences, predicted_classes): # 修改重复的类别名
                    # 给相同类别的标签添加数字后缀
                    if class_name not in class_count:
                        class_count[class_name] = 0
                    class_count[class_name] += 1

                    #创建新标签
                    new_class_name = f"{class_name}{class_count[class_name] - 1}" if class_count[class_name] > 1 else class_name
                    new_predicted_classes.append(new_class_name)
                    pp = '(' + ', '.join([str(item) for item in box]) + ')'
                    self.sum += 1
                    self.count += 1
                    # 向TableView添加信息
                    row = []
                    file=f'{self.save_dir}/{frame_filename}/crops/{new_class_name}/{new_class_name}.jpg'
                    row.append(QStandardItem(str(self.count)))
                    row.append(QStandardItem(str(file)))
                    row.append(QStandardItem(str(self.sum)))
                    row.append(QStandardItem(str(new_class_name)))
                    row.append(QStandardItem(str(f'{confidences:.2f}')))
                    row.append(QStandardItem(str(pp)))
                    self.tb_model.appendRow(row)
                self.fileName1 = file_path
                self.comboBox.clear()
                self.comboBox.addItem('全部')
                self.comboBox.addItems(new_predicted_classes)
                self.image_boBox[file_path] = new_predicted_classes
                self.image_sum[file_path] = self.sum

            
                self.label_6.setText(str(self.sum)) # 总目标数
                # self.label_8.setText(str(round(tm*1000, 3))+'ms') # 用时
                self.label_8.setText(tm + 'ms')
                QApplication.processEvents()  # 让事件循环及时处理UI更新

        else:
            self.predictor.paused = not self.predictor.paused

    def save_detect_frame(self):
        # 获取目标文件夹路径
        folder = QFileDialog.getExistingDirectory(self, "选择目标文件夹")
        if not os.path.exists(self.save_dir):
            QMessageBox.warning(self, "错误", "源文件夹不存在!")
            return
        
        if not os.path.exists(folder):
            QMessageBox.warning(self, "错误", "目标文件夹不存在！")
            return
        
        # 计算源文件夹中的总文件数
        self.total_files = len([f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))])
        if self.total_files == 0:
            QMessageBox.warning(self, "错误", "源文件夹没有文件可复制！")
            return
        # 初始化进度条
        self.progress_bar.setMaximum(self.total_files)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.copied_files = 0

        # 遍历源文件夹并复制文件
        try:
            for filename in os.listdir(self.save_dir):
                source_path = os.path.join(self.save_dir, filename)
                target_path = os.path.join(folder, filename)

                # 如果是文件，则复制
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, target_path)
                    self.copied_files += 1
                    self.progress_bar.setValue(self.copied_files)  # 更新进度条
                    QApplication.processEvents()  # 处理UI事件，保持进度条更新

            QMessageBox.information(self, "完成", "文件已成功复制！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"复制过程中出错：{e}")
        finally:
            self.progress_bar.hide()
            



    def predict(self):
        self.cfg.model = self.cfg.model or "yolov8n.pt"
        self.cfg.imgsz = check_imgsz(self.cfg.imgsz, min_dim=2)  # check image size
        self.cfg.source = self.fileName1 #self.cfg.source if self.cfg.source is not None else ROOT / "assets"
        self.predictor = DetectionPredictor(self.cfg)
        self.save_dir = self.predictor.save_dir
        results = self.predictor.predict_cli()
        return results
    
    def write_one_results(self, idx, det, im0 ):
        im0 = im0.copy()
        self.annotator0 = Annotator(im0, line_width=self.cfg.line_thickness, example=str(self.model.names))

        # write
        num = 0
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in det:
            
            c = int(cls)  # integer class
            # Add bbox to image
            num += 1
            if num == idx:
                label = None if self.cfg.hide_labels else (
                    self.model.names[c] if self.cfg.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                self.annotator0.box_label(xyxy, label, color=colors(c, True))
                # 目标选择之后添加信息
                self.label_11.setText(self.model.names[c])
                self.label_13.setText(f'{conf:.2f}')
                self.label_19.setText(str(xyxy[0]))
                self.label_20.setText(str(xyxy[1]))
                self.label_21.setText(str(xyxy[2]))
                self.label_22.setText(str(xyxy[3]))
        return self.annotator0.result()
    
    def exit_app(self):
        if self.predictor:
            self.predictor.stop = True
        # 弹出确认对话框
        reply = QMessageBox.question(
            self, 
            "确认退出", 
            "确定要退出吗？", 
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            QApplication.instance().quit()  # 调用 QApplication 的 quit 方法退出程序



class DetectionPredictor(BasePredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    


    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        if self.return_outputs:
            self.output["det"] = det.cpu().numpy()

        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string
    
    
        
    

# 使用Hydra装饰器
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def main(cfg):
    app = QApplication(sys.argv)
    window = main0(cfg)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
    