#将图像转换为测试向量：把一个32x32的二进制图像矩阵转换为1x1024的向量
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
#加载训练集到大矩阵trainingMat  
    hwLabels = []  
    trainingFileList = listdir('trainingDigits')           #os模块中的listdir('str')可以读取目录str下的所有文件名，返回一个字符串列表  
    m = len(trainingFileList)  
    trainingMat = zeros((m,1024))  
    for i in range(m):  
        fileNameStr = trainingFileList[i]                  #训练样本的命名格式：1_120.txt  
        fileStr = fileNameStr.split('.')[0]                #string.split('str')以字符str为分隔符切片，返回list，这里去list[0],得到类似1_120这样的  
        classNumStr = int(fileStr.split('_')[0])           #以_切片，得到1，即类别  
        hwLabels.append(classNumStr)  
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)  
          
    #逐一读取测试图片，同时将其分类     
    testFileList = listdir('testDigits')         
    errorCount = 0.0  
    mTest = len(testFileList)  
    for i in range(mTest):  
        fileNameStr = testFileList[i]  
        fileStr = fileNameStr.split('.')[0]       
        classNumStr = int(fileStr.split('_')[0])  
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)  
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)  
        if (classifierResult != classNumStr): errorCount += 1.0  
    print ("\nthe total number of errors is: %d" % errorCount)  
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest))</span>)  
