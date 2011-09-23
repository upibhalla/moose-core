import sys
from PyQt4 import QtGui,QtCore
from PyQt4.Qt import Qt
import pickle
from header import newGLWindow
from oglfunc.sobjects import *

def updateColor(w,q,t):
    for i in range(len(w.vizObjects)):
        w.vizObjects[i].r,w.vizObjects[i].g,w.vizObjects[i].b = w.colorMap[q[t][i]-1]
    w.updateGL()

def getValue(value):
    updateColor(w,q,value)
    
def playMovie():
    t = newWin.slider.value()
    t += 1        
    if t < len(q)-1:
        newWin.slider.setValue(t)
    else:
        stopTimer()
        
# def playMovie1():
#     t = newWin.slider.tickPosition()
#     while t<(len(q)-1):
#         time.sleep(0.2)
#         newWin.slider.setValue(t)
#         #w.rotate([1,0,0],50)
#         w.rotate([0,1,1],1)
#         w.translate([0,0.05,0])
#         updateColor(w,q,t)
#         pic = w.grabFrameBuffer()
#         pic.save('movie/sim_'+str(t)+'.png','PNG')
#         t += 1
#         #w.rotate([1,0,0],-50)

def startTimer():
    newWin.ctimer.start(200)

def stopTimer():
    newWin.ctimer.stop()

app = QtGui.QApplication(sys.argv) 
newWin = newGLWindow()
newWin.show()
newWin.setWindowState(Qt.WindowMaximized)

f = open(sys.argv[1],'r')
q = pickle.load(f)
f.close()

#timer
newWin.ctimer = QtCore.QTimer()
newWin.connect(newWin.ctimer, QtCore.SIGNAL("timeout()"),playMovie)

#toolbar
newWin.toolbar = QtGui.QToolBar()
newWin.toolbar.setMinimumHeight(30)
newWin.addToolBar(Qt.BottomToolBarArea,newWin.toolbar)

#play
newWin.playButton = QtGui.QToolButton(newWin.toolbar)
newWin.playButton.setIcon(QtGui.QIcon('play.png'))
newWin.playButton.setGeometry(0,0,30,30)
newWin.connect(newWin.playButton, QtCore.SIGNAL('clicked()'),startTimer)

#pause
newWin.pauseButton = QtGui.QToolButton(newWin.toolbar)
newWin.pauseButton.setIcon(QtGui.QIcon('pause.png'))
newWin.pauseButton.setGeometry(30,0,30,30)
newWin.connect(newWin.pauseButton, QtCore.SIGNAL('clicked()'),stopTimer)

#slider
newWin.slider = QtGui.QSlider(QtCore.Qt.Horizontal,newWin.centralwidget)
newWin.slider.setRange(0,len(q)-2)
newWin.slider.setTickPosition(0)
newWin.connect(newWin.slider, QtCore.SIGNAL('valueChanged(int)'), getValue)

#default setting
#newWin.verticalLayout.addWidget(newWin.toolbar)
newWin.verticalLayout.addWidget(newWin.slider)
newWin.setWindowState(Qt.WindowMaximized)

w =  newWin.mgl
cnf = q[len(q)-1]
for i in range(len(cnf)):
    a = locals()[cnf[i][0]](w,cnf[i][1],cnf[i][2])
    a.setCellParentProps(cnf[i][3],cnf[i][4],1,0,0)
    w.vizObjects.append(a)
w.setColorMap()
w.updateGL()
         
sys.exit(app.exec_())
