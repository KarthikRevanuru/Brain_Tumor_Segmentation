with open("Histogram.txt") as f:
    f.readline()
    hashMap = {}
    imgNames = []
    for lines in f.readlines():
	#print (lines)
        [name,X,Y,Z,Label] = lines.split()[:5]
        X,Y,Z,Label = int(X),int(Y),int(Z),int(Label)
        try:
            hashMap[name].append([X,Y,Z,Label])
        except:
            hashMap[name] = [[X,Y,Z,Label]]
            imgNames.append(name)
                
    for name in imgNames:
        arrVal = hashMap[name]
        minX = -1
        minY = -1
        minZ = -1
        maxX = -1
        maxY = -1
        maxZ = -1
        for label in xrange(240):
            X,Y,Z,tag =  arrVal[label]
            if(tag == 1):
                if(label<237 and (arrVal[label+1][3] == 1 or arrVal[label+2][3] == 1 or arrVal[label+3][3] == 1)): 
                    minX = X
                    break
                
        for label in xrange(240,480):
            X,Y,Z,tag = arrVal[label]
            if(tag == 1):
                if(label<477 and (arrVal[label+1][3] == 1 or arrVal[label+2][3] == 1 or arrVal[label+3][3] == 1)): 
                    minY = Y
                    break 
                       
        for label in xrange(480,len(arrVal)):
            X,Y,Z,tag = arrVal[label]
            if(tag == 1):
                if(label<len(arrVal)-3 and (arrVal[label+1][3] == 1 or arrVal[label+2][3] == 1 or arrVal[label+3][3] == 1)): 
                    minZ = Z
                    break
        ###Maximum
        for label in xrange(239,-1,-1):
            X,Y,Z,tag =  arrVal[label]
            if(tag == 1):
                if(label>3 and (arrVal[label-1][3] == 1 or arrVal[label-2][3] == 1 or arrVal[label-3][3] == 1)): 
                    maxX = X
                    break
                
        for label in xrange(479,239,-1):
            X,Y,Z,tag = arrVal[label]
            if(tag == 1):
                if(label>242 and (arrVal[label-1][3] == 1 or arrVal[label-2][3] == 1 or arrVal[label-3][3] == 1)): 
                    maxY = Y
                    break 
                       
        for label in xrange(len(arrVal)-1,479,-1):
            X,Y,Z,tag = arrVal[label]
            if(tag == 1):
                if(label>482 and (arrVal[label-1][3] == 1 or arrVal[label-2][3] == 1 or arrVal[label-3][3] == 1)): 
                    maxZ = Z
                    break
        #TODO Change output according to needs
        #Spaces instead of Tabs for indentation         
        print name,minX,minY,minZ,maxX,maxY,maxZ        
