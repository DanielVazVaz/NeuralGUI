## ====================================================================== ##
## ---------------- GUI para una Red Neuronal --------------------------- ##
## ====================================================================== ##

'''
En este programa, utilizamos una interfaz para ubicar y determinar una red 
neuronal a partir de un archivo de datos proporcionado en formato .csv. Para
ello, creamos una interfaz que nos permita abrir un diálogo y leer los datos 
de dicho archivo.

Es un trabajo en proceso y seguramente pete más veces que no.
''' 

## ------------------------- IMPORTES ----------------------------------- ##
import numpy as np
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMenu, 
    QGridLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, 
    QSpacerItem, QLabel, QTextEdit, QVBoxLayout, QComboBox, QLineEdit, 
    QFileDialog, QFrame, QGroupBox) 
from PyQt5.QtGui import QIcon, QTextCursor, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PaquetesNeural.NN_Master import HiddenLayers, NeuralNetwork
from PaquetesNeural.lfunctions import (sigmoid, relu, error, lin, soft, 
    crossEntrop)
from pandas import read_excel
## ---------------------------------------------------------------------- ##


## ------------------------- GUI ---------------------------------------- ##
app = None
plt.close('all')

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.showUI()
        self.show()

    def initUI(self):
        self.setGeometry(100, 50, 850, 600)
        self.setWindowTitle('Neural Network Interface Solver')
        self.setWindowIcon(QIcon('brainGUI.png'))
        self.toolbar = self.addToolBar('Toolbar')
        self.functions = [sigmoid, sigmoid, sigmoid]
        self.erf = error

    def showUI(self):
# Central widget and layout
        wid = QWidget()
        self.setCentralWidget(wid)
        grid = QGridLayout()
        wid.setLayout(grid)
        grid.setSpacing(5)

# Creamos las cuatro cajas que vamos a usar
        Caja1 = QGroupBox('Dimensión de los datos del entrenamiento')
        Caja2 = QGroupBox('Área de la representación del error')
        Caja3 = QGroupBox('Parámetros de la red neuronal')
        Caja4 = QGroupBox('Zona de resultados')
        Caja5 = QGroupBox('Zona de botones')
        Caja1.setObjectName('PrimeraCaja')

# Creamos los cuatro layouts. Son 4 Grids, al fin y al cabo.
        Caja1Layout = QGridLayout()
        Caja2Layout = QGridLayout()
        Caja3Layout = QGridLayout()
        Caja4Layout = QGridLayout()
        Caja5Layout = QGridLayout()

        Caja1.setLayout(Caja1Layout)
        Caja2.setLayout(Caja2Layout)
        Caja3.setLayout(Caja3Layout)
        Caja4.setLayout(Caja4Layout)
        Caja5.setLayout(Caja5Layout)

        grid.addWidget(Caja1,0,0,1,2) # 1 Fila - 2 Columnas
        grid.addWidget(Caja2,1,0,1,4) # 1 Fila - 4 Columnas
        grid.addWidget(Caja3,2,0,1,4) # 1 Fila - 2 Columnas
        grid.addWidget(Caja4,0,4,2,1) # 2 Filas - 1 Columna
        grid.addWidget(Caja5, 3,0,3,4) # 3 Filas - 4 Columnas


        grid.setColumnStretch(0,2)
        Caja1.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum) # Con esto se consigue dejar pequeña la mierda esa
        Caja3.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)



# Objects
        # Plot object
        self.figure = plt.figure(tight_layout = True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(200,200)  # Esto hace que no se pueda hacer el widget to pequeño
        ax = self.figure.add_subplot(111)
        plt.xlabel('Iteración')
        plt.ylabel('Error')
        plt.title('Error vs Iteración')
        self.canvas.draw()
        # Labels for the functions
        label1 = QLabel('Input - H1')
        label2 = QLabel('H1 - H(-1)')
        label3 = QLabel('H(-1) - Output')
        label4 = QLabel('Error')
        label5 = QLabel('Layers'); label6 = QLabel('Neurons'); label7 = QLabel('Learning rate'); label8 = QLabel('Regularization')
        label9 = QLabel('Resultados:')
        label10 = QLabel('Dimensión de inputs')
        label11 = QLabel('Dimensión de outputs')
        # Selectors for the functions
        self.combo1 = QComboBox(self)
        self.combo1.addItem('Sigmoid')
        self.combo1.addItem('ReLu')
        self.combo1.addItem('Linear')
        self.combo1.addItem('SoftMax')
        self.combo2 = QComboBox(self)
        self.combo2.addItem('Sigmoid')
        self.combo2.addItem('ReLu')
        self.combo2.addItem('Linear')
        self.combo2.addItem('SoftMax')
        self.combo3 = QComboBox(self)
        self.combo3.addItem('Sigmoid')
        self.combo3.addItem('ReLu')
        self.combo3.addItem('Linear')
        self.combo3.addItem('SoftMax')
        self.combo4 = QComboBox(self)
        self.combo4.addItem('Quadratic Error')
        self.combo4.addItem('Cross-Entropy function')
        # Botones
        btn = QPushButton('Cargar parámetros')
        btn.clicked.connect(self.checkParameters)
        btn2 = QPushButton('Crear Redes Neuronales')
        btn2.clicked.connect(self.initNetwork)
        btn3 = QPushButton('Resolver red')
        btn3.clicked.connect(self.solveNetwork)
        btn3.setStyleSheet("font: bold; background-color: cyan; font-size: 26px; border-width: 100px; border-color: black")
        btn3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn4 = QPushButton('Cargar Inputs y Outputs')
        btn4.clicked.connect(self.readInputs)
        # Edit Texts
        self.ed1 = QLineEdit(' '); self.ed2 = QLineEdit(' '); self.ed3 = QLineEdit(' '); self.ed4 = QLineEdit(' ')
        self.ed1.setText('1');self.ed2.setText('1');self.ed3.setText('1');self.ed4.setText('1')
        self.ed5 = QLineEdit('1'); self.ed6 = QLineEdit('1')
        self.ed5.setMaximumWidth(30); self.ed6.setMaximumWidth(30)
        # Bloque de texto
        self.block = QTextEdit('Status:')
        self.block.moveCursor(QTextCursor.End)
        self.block.insertPlainText('\n' + 'Run for results')
        self.block.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# Colocamos los objetos
        # Labels
        Caja1Layout.addWidget(label10, 0,0); Caja1Layout.addWidget(label11,1,0)
        Caja1Layout.addWidget(self.ed5,0,1); Caja1Layout.addWidget(self.ed6,1,1)
        Caja1Layout.setColumnStretch(1,3)
        # Plot
        Caja2Layout.addWidget(self.canvas,0,0) # 1 fila, 4 columnas
        # Parametros a introducir
        Caja3Layout.addWidget(label1, 0,0); Caja3Layout.addWidget(self.combo1, 0,1) 
        Caja3Layout.addWidget(label2, 1,0); Caja3Layout.addWidget(self.combo2, 1,1)
        Caja3Layout.addWidget(label3, 2,0); Caja3Layout.addWidget(self.combo3, 2,1)
        Caja3Layout.addWidget(label4, 3,0); Caja3Layout.addWidget(self.combo4, 3,1)
        Caja3Layout.addWidget(label5, 0,2); Caja3Layout.addWidget(self.ed1,0,3)
        Caja3Layout.addWidget(label6, 1,2); Caja3Layout.addWidget(self.ed2,1,3)
        Caja3Layout.addWidget(label7, 2,2); Caja3Layout.addWidget(self.ed3,2,3)
        Caja3Layout.addWidget(label8, 3,2); Caja3Layout.addWidget(self.ed4,3,3)
        # Bloque de texto
        Caja4Layout.addWidget(label9, 0,0,1,1)
        Caja4Layout.addWidget(self.block, 1,0,1,1)
        # Botones
        Caja5Layout.addWidget(btn, 0,0); Caja5Layout.addWidget(btn2, 2,0)
        Caja5Layout.addWidget(btn3,0,1,3,3)
        Caja5Layout.addWidget(btn4, 1,0)


    def checkParameters(self):
        f1s = str(self.combo1.currentText())
        f2s = str(self.combo2.currentText())
        f3s = str(self.combo3.currentText())
        f4s = str(self.combo4.currentText())
        if f1s[:2] == 'Si':
            f1 = sigmoid
        elif f1s[0] =='R':
            f1 = relu
        elif f1s[0] =='L':
            f1 = lin
        elif f1s[:2] == 'So':
            f1 = soft

        if f2s[:2] == 'Si':
            f2 = sigmoid
        elif f2s[0] =='R':
            f2 = relu
        elif f2s[0] =='L':
            f2 = lin
        elif f2s[:2] == 'So':
            f2 = soft

        if f3s[:2] == 'Si':
            f3 = sigmoid
        elif f3s[0] =='R':
            f3 = relu
        elif f3s[0] =='L':
            f3 = lin
        elif f3s[:2] == 'So':
            f3 = soft

        if f4s[:2] == 'Qu':
            f4 = error
        elif f4s[0] =='C':
            f4 = crossEntrop


        self.functions = [f1, f2, f3]
        self.erf = f4
        self.NI = int(self.ed5.text())
        self.NO = int(self.ed6.text())
        self.NL = int(self.ed1.text()); self.NN = int(self.ed2.text()); self.lr = float(self.ed3.text())
        self.reg = float(self.ed4.text())
        print('The functions are: ', self.functions)
        print('The error calculus is: ', self.erf)
        print('The dimension of the inputs is: ', self.NI)
        print('The dimension of the outputs is: ', self.NO)
        print('The number of layers is: ', self.NL)
        print('The number of neurons per hidden layer is: ', self.NN)
        print('The learning rate is: ', self.lr)
        print('The reg parameter is: ', self.reg)


    def readInputs(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File')
        Inputs_df = read_excel(fname[0])
        self.Inputs = np.array(Inputs_df.iloc[:,:self.NI])
        self.Outputs = np.array(Inputs_df.iloc[:,self.NI:])
        print(self.Inputs.shape)
        print(self.Outputs.shape)
        if self.NO != self.Outputs.shape[1]:
            print('AVISO: EL NUMERO DE OUTPUTS NO COINCIDE CON LOS DATOS DADOS')



    def initNetwork(self):
        self.Layers = HiddenLayers(self.functions[0], self.functions[1], self.functions[2])
        self.Layers.setlayers(self.Inputs.shape[1], self.Outputs.shape[1], self.NL, self.NN)
        print(self.Layers.f)

    def solveNetwork(self):
        self.block.clear()
        self.calculo = CalculoNeural(NeuralNetwork(self.Inputs, self.Layers.W, self.Layers.b, self.Layers.f, self.lr, self.Outputs), self.erf, self.reg)
        self.calculo.mySignal.connect(self.setText) # Se conecta el objeto de señal con el slot de setText, definido abajo
        # Creamos las listas para guardar los errores
        self.IterPlot = []
        self.ErrorPlot = []
        self.calculo.PlotSignal.connect(self.setPlot) # Lo mismo. Se conecta la señal PlotSignal con el slot de setPlot.
        self.calculo.start()

    def setText(self, text):  # Este text es lo que se envia con la señal
        self.block.moveCursor(QTextCursor.End)
        self.block.insertPlainText('\n' + text)

    def setPlot(self, value, iteration):
        print('Iteración: ', iteration, ' Valor del error : ', value)
        if iteration > 1:
            self.IterPlot.append(iteration/10000)
            self.ErrorPlot.append(value)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if iteration > 10000:
            ax.plot(self.IterPlot, self.ErrorPlot,'-*r')
        plt.xlabel('Iteración')
        plt.ylabel('Error')
        plt.title('Error vs Iteración')
        self.canvas.draw()



## ------------ THREAD PARA EL CALCULO DE LA NEURAL NETWORK ------------- ##
class CalculoNeural(QThread):  # Este QThread nos permite que no se pete la gui mientras anda
    mySignal = pyqtSignal(type(' '))   # Señal que llevará un string con el error
    PlotSignal = pyqtSignal(type(3.32), type(2)) # Señal que llevará el valor del error. Con un Integer que será la iteración.
    def __init__(self, neural, erf, reg):
        QThread.__init__(self)
        self.NeuralNetwork = neural
        self.erf = erf   # Hay que meterlo como otro argumento porque soy idiota y en la función no está
        self.reg = reg   # Lo mismo que lo que pone arriba

    def run(self):
        for j in range(200000):
            self.NeuralNetwork.Forward_Prop()
            self.NeuralNetwork.Calculate_Error(self.erf)
            self.NeuralNetwork.Backward_Prop()
            self.NeuralNetwork.Actualize(self.reg)
            if j%10000 ==0:
                self.mySignal.emit(str(self.NeuralNetwork.ErrorTotal)[:7])
                self.PlotSignal.emit(self.NeuralNetwork.ErrorTotal, j)
        self.NeuralNetwork.Forward_Prop()
        self.NeuralNetwork.Calculate_Error(self.erf)
        self.mySignal.emit('The final error is: ' + str(self.NeuralNetwork.ErrorTotal)[:7])
        self.PlotSignal.emit(self.NeuralNetwork.ErrorTotal, j)
## ====================================================================== ##



## ------------------------- INICIALIZACIÓN ----------------------------- ##
def main():
    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    else:
        print('QApplication instance already exists: %s' % str(app))
    w = App()
    w.setStyleSheet('''
                    QGroupBox{font-family: Times New Roman; font-size: 17px;
                    color: red; border-color: black}
                    QGroupBox#PrimeraCaja{color: blue;}
                    QLabel{font-family: Times New Roman; font-size: 10px;}
                    
                    ''')
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
## ---------------------------------------------------------------------- ##