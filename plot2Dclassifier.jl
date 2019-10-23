using PyCall
using PyPlot

function plot2Dclassifier(X,y,model;proba=false)

	increment = 100

	figure()
	plot(X[y.==-1,1],X[y.==-1,2],"bo")
	plot(X[y.==1,1],X[y.==1,2],"r^")

	(xmin,xmax) = xlim()
	xDomain = range(xmin,stop=xmax,length=increment)
	(ymin,ymax) = ylim()
	yDomain = range(ymin,stop=ymax,length=increment)

	xValues = repeat(xDomain,1,length(xDomain))
	yValues = repeat(yDomain',length(yDomain),1)

	z = model.predict([xValues[:] yValues[:]])
	if proba
		z = model.predict_proba([xValues[:] yValues[:]])[:,end]
	end

	@assert(length(z) == length(xValues),"Size of model function's output is wrong");

	zValues = reshape(z,size(xValues))

	if all(zValues[:] == 1)
    		cm = [(0,0,.5)];
	elseif all(zValues[:] == -1)
    		cm = [(.5,0,0)];
	else
    		cm = [(0,0,.5);(.5,0,0)];
	end
	matcolors = pyimport("matplotlib.colors")
	
	if proba
		cs = contourf(xValues, yValues, zValues, levels=15, cmap="RdBu_r")
		colorbar(cs)
	else
		cmap = matcolors.ListedColormap(cm,"A")
		contourf(xValues,yValues,zValues,cmap=cmap)
	end
end
