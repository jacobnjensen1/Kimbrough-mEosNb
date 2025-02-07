import numpy as np
from matplotlib import colors
import mpl_scatter_density
import matplotlib.pyplot as plt
# import plotly.express as px
import warnings
import seaborn as sns
import fcsparser
import pandas as pd
from matplotlib.path import Path

#This is largely copied from /home/jj2765/DAmFRET_denoising/tomato/DAmFRETClusteringTools/ClusterPlotting.py, but with modifications desired by Hannah.


#add arial
from matplotlib import font_manager
arialPath = "/n/projects/jj2765/mEos_nanobody/fonts/arial.ttf"
font_manager.fontManager.addfont(arialPath)

# Set the default font size and family
#does this change the defaults for plots that don't use this module, but do import it?
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

def plotDAmFRETCluster(x, y, color="blue", ax=None, title=None, vmin=0.25, xlab="mEos3 concentration (p.d.u.)", ylab="AmFRET"):
    """
    Inspired by Tayla's function and the "double" example from the mpl_scatter_density README
    If ax is not given, the only plot will be the cluster plot
    """
    if not ax:
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(1,1,1, projection= 'scatter_density')

    #     ax.scatter_density(x, y, color=color) #only shows most dense part
    #     ax.scatter_density(x, y, color=color, norm=colors.LogNorm()) #blocky
    #setting dpi in scatter_density the same as figure dpi leads to swirls and other fun problems.
    if len(x) >= 2: #only one cell breaks things because vmin = vmax
        ax.scatter_density(x, y, color=color, norm=colors.LogNorm(), vmin=vmin, vmax=np.nanmax, dpi=100, clip_on=True) #I think this looks good, clip_on added on 6/13 to better address saturated acceptor values
    #update: clip_on doesn't actually help.
    #increasing zorder allows the plot to lay over the spines, but that shouldn't be necessary, the spines should be outside of the data.
    #Because they aren't actually outside of the data, I set the right spine to be 0.5 further out. That seems to be the width of a point in scatter_density, but I'm not sure if it's flexible across dpis and figsizes.
    #ax.scatter_density(x, y, color=color, norm=colors.LogNorm(), vmin=vmin, vmax=np.nanmax, dpi=100, zorder=5)
    #ax.scatter_density(x, y, color=color, norm=colors.LogNorm(), vmin=vmin, vmax=np.nanmax, dpi=None, zorder=5)

    ax.spines.right.set_position(("outward", 0.5))
    ax.set_xlabel(xlab, loc="center")
    ax.set_ylabel(ylab, loc="center")
    if title:
        ax.set_title(title)
    return ax

def plotDAmFRETDensity(x, y, ax=None, vmin=0.25, figWidth=4.5, figHeight=3, title=None, xlab="mEos3 concentration (p.d.u.)", ylab="AmFRET", returnFig=False, xlims=None, ylims=None, logX=False, mpl_dpi=100, cmap=None):
    """
    Generates a density plot for all cells in a well.
    Based on Tayla's function.
    If ax is not given, the only plot will be the density plot
    xlims and ylims must be a tuple of (min, max), or will be automatically determined if None
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #I've only ever seen warnings in mpl_scatter_density due to empty bins, which we expect
        if not ax:
            fig = plt.figure(figsize=(figWidth, figHeight), dpi=150)
            ax = fig.add_subplot(1,1,1, projection= 'scatter_density')
        if xlims is None:
          xlims = (min(x), max(x))
        if ylims is None:
            ylims = (min(y), max(y))
            if type(y).__name__ == "Series":
                if y.name == "AmFRET":
                    ylims = (-0.2,2) #for this paper, use this AmFRET range by default
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
    
        norm = colors.LogNorm()
        if cmap is None:
            cmap = plt.get_cmap("viridis")
            cmap.set_under(alpha=0)
    
        ax.scatter_density(x, y, norm=norm, vmin=vmin, vmax=np.nanmax, cmap=cmap, dpi=mpl_dpi, clip_on=True) #BAD
        #ax.scatter_density(x, y, norm=norm, vmin=vmin, vmax=np.nanmax, cmap=cmap, dpi=100, zorder=5)
        #ax.scatter_density(x, y, norm=norm, vmin=vmin, vmax=np.nanmax, cmap=cmap, dpi=None, zorder=5)
    
        ax.spines.right.set_position(("outward", 0.5))
        ax.set_xlabel(xlab, loc="center")
        ax.set_ylabel(ylab, loc="center")
        if title:
            ax.set_title(title)
    
        yTicks = np.arange(-10,15.1, 0.5) #way larger range than will likely ever be seen
        #values are -10, -9.5, -9, ... but only ticks within the AmFRET bounds will be shown
        #selecting within range isn't necessary because we set ylim after this
        ax.set_yticks(yTicks)
    
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
    
    
    
        if logX:
            ax.set_xscale("log")
    
        #if y contains a pd.Series and y axis is AmFRET, add the dashed line at 0
        #checks type because y could be a list or other non pd.Series type.
        #This could fail if a Series in a different module is provided, but I think that's unlikely
        #Even polars series seem to use the <Series>.name convention
        if type(y).__name__ == "Series":
            if y.name == "AmFRET":
                ax.hlines(0, xlims[0], xlims[1], color=(0.6,0.6,0.6), linestyle="--")
        
        if returnFig:
          return fig
        return ax

def plotDAmFRETClusters(x, y, labels, ax=None, colors=plt.get_cmap("tab10").colors, vmin=0.25, figWidth=4.5, figHeight=3, title=None, xlab="mEos3 concentration (p.d.u.)", ylab="AmFRET", returnFig=False, xlims=None, ylims=None, logX=False, labelOrder=None, addPopulationStats=True):
    """
    Generates a single figure with subplots of the whole well density and cluster densities.
    colors takes an iterable of colors (names or iterable of rgb)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #I've only ever seen warnings in mpl_scatter_density due to empty bins, which we expect

        if not ax:
            fig = plt.figure(figsize=(figWidth,figHeight), dpi=150)
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

        uniqueLabels, labelCounts = np.unique(labels, return_counts=True)
        labelCountDict = dict(zip(uniqueLabels, labelCounts))

        #start positions - will be relative to size of axis, not data
        textX = 0.01
        textYMax = 0.98

        if labelOrder is None:
            #make the label for the highest pop
            label = uniqueLabels.max()
            plotDAmFRETCluster(x[labels == label], y[labels == label], colors[label], ax, vmin=vmin, xlab=xlab, ylab=ylab)
            if addPopulationStats:
                higherText = ax.text(textX, textYMax, f"{labelCountDict[label]}, {labelCountDict[label] / sum(labelCountDict.values()):.2f}", va="top", ha="left", color=colors[label], transform = ax.transAxes)
            
            #make label for non-highest pop(s)
            for label in uniqueLabels[-2::-1]:
                plotDAmFRETCluster(x[labels == label], y[labels == label], colors[label], ax, vmin=vmin, xlab=xlab, ylab=ylab)

                if addPopulationStats:
                    higherText = ax.annotate(f"{labelCountDict[label]}, {labelCountDict[label] / sum(labelCountDict.values()):.2f}", xycoords=higherText, xy=(0,-1), color=colors[label], horizontalalignment="left", transform = ax.transAxes)

        else:
            label = labelOrder[0]
            
            plotDAmFRETCluster(x[labels == label], y[labels == label], colors[label], ax, vmin=vmin, xlab=xlab, ylab=ylab)
            if addPopulationStats:
                higherText = ax.text(textX, textYMax, f"{labelCountDict[label]}, {labelCountDict[label] / sum(labelCountDict.values()):.2f}", va="top", ha="left", color=colors[label], transform = ax.transAxes)
            
            #make label for non-highest pop(s)
            for label in labelOrder[1:]:
                plotDAmFRETCluster(x[labels == label], y[labels == label], colors[label], ax, vmin=vmin, xlab=xlab, ylab=ylab)

                if addPopulationStats:
                    higherText = ax.annotate(f"{labelCountDict[label]}, {labelCountDict[label] / sum(labelCountDict.values()):.2f}", xycoords=higherText, xy=(0,-1), color=colors[label], horizontalalignment="left", transform = ax.transAxes)
        
        if xlims is None:
            xlims = (min(x), max(x))
        if ylims is None:
            ylims = (min(y), max(y))
            if type(y).__name__ == "Series":
                if y.name == "AmFRET":
                    ylims = (-0.2,2) #for this paper, use this AmFRET range by default
                    
        yTicks = np.arange(-10,15.1, 0.5) #way larger range than will likely ever be seen
        #values are -10, -9.5, -9, ... but only ticks within the AmFRET bounds will be shown
        #selecting within AmFRET range isn't necessary because we set ylim after this
        ax.set_yticks(yTicks)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        if logX:
            ax.set_xscale("log")

        #if y contains a pd.Series and y axis is AmFRET, add the dashed line at 0
        #checks type because y could be a list or other non pd.Series type.
        #This could fail if a Series in a different module is provided, but I think that's unlikely
        #Even polars series seem to use the <Series>.name convention
        if type(y).__name__ == "Series":
            if y.name == "AmFRET":
                ax.hlines(0, xlims[0], xlims[1], color=(0.6,0.6,0.6), linestyle="--")

        if title:
            ax.set_title(title)

        #fig.tight_layout()
        if returnFig:
            return fig
        return ax

def plotDAmFRETDensityAndClusters(x, y, labels, colors=plt.get_cmap("tab10").colors, vmin=0.25, figWidth=9, figHeight=3, title=None, xlab="mEos3 concentration (p.d.u.)", ylab="AmFRET", xlims=None, ylims=None, logX=False, labelOrder=None):
    """
    Generates a single figure with subplots of the whole well density and cluster densities.
    colors takes an iterable of colors (names or iterable of rgb)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #I've only ever seen warnings in mpl_scatter_density due to empty bins, which we expect

        fig = plt.figure(figsize=(figWidth,figHeight), dpi=150)
        ax = fig.add_subplot(1, 2, 1, projection='scatter_density')
        plotDAmFRETDensity(x, y, ax, vmin=vmin, xlab=xlab, ylab=ylab, xlims=xlims, ylims=ylims, logX=logX)

        if xlims is None:
            xlims = (min(x), max(x))
        if ylims is None:
            ylims = (min(y), max(y))
            if type(y).__name__ == "Series":
                if y.name == "AmFRET":
                    ylims = (-0.2,2) #for this paper, use this AmFRET range by default
                    
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        ax = fig.add_subplot(1, 2, 2, projection='scatter_density')
        plotDAmFRETClusters(x,y,labels, ax, colors, vmin=vmin, xlab=xlab, ylab=ylab, xlims=xlims, ylims=ylims, logX=logX, labelOrder=labelOrder)

        #if y contains a pd.Series and y axis is AmFRET, add the dashed line at 0
        #checks type because y could be a list or other non pd.Series type.
        #This could fail if a Series in a different module is provided, but I think that's unlikely
        #Even polars series seem to use the <Series>.name convention
        if type(y).__name__ == "Series":
            if y.name == "AmFRET":
                ax.hlines(0, xlims[0], xlims[1], color=(0.6,0.6,0.6), linestyle="--")

        if title:
            fig.suptitle(title)

        fig.tight_layout()
        return fig

def plotBDFPAcceptorContours(dataToPlot, labelColumn="manualLabel", colors=plt.get_cmap("tab10").colors, ax=None, xlims=None, ylims=None, returnFig=False):
    if not ax:    
        fig = plt.figure()
        ax = plt.gca()

    if "BDFP/SSC" not in dataToPlot:
        dataToPlot["BDFP/SSC"] = dataToPlot["BDFP1.6-A"] / dataToPlot["SSC 488/10-A"]

    colorsToUse = {label: colors[label] for label in dataToPlot[labelColumn].unique()}

    #palette can take a dictionary, so this should work
    sns.kdeplot(dataToPlot, x="Acceptor/SSC", y="BDFP/SSC", hue=labelColumn, palette=colorsToUse, fill=True, alpha=0.5, legend=False, log_scale=True, ax=ax, clip=(xlims, ylims))

    defaultXLabel = "mEos3 concentration (p.d.u.)"
    defaultYLabel = "BDFP1.6:1.6 concentration (p.d.u.)"

    ax.set_xlabel(defaultXLabel)
    ax.set_ylabel(defaultYLabel)
    
    if returnFig:
        return fig

def plotBDFPAcceptorContours_test(dataToPlot, labelColumn="manualLabel", colors=plt.get_cmap("tab10").colors, ax=None, xlims=None, ylims=None, returnFig=False, hueOrder=None, alpha=np.linspace(0.2,0.6,9), title=None):
    if not ax:    
        fig = plt.figure()
        ax = plt.gca()

    if "BDFP/SSC" not in dataToPlot:
        dataToPlot["BDFP/SSC"] = dataToPlot["BDFP1.6-A"] / dataToPlot["SSC 488/10-A"]

    colorsToUse = {str(label): colors[label] for label in dataToPlot[labelColumn].unique()}
    if hueOrder is not None:
        filteredHueOrder = [label for label in hueOrder if label in dataToPlot[labelColumn].unique()]
        stringHueOrder = [str(label) for label in filteredHueOrder]
    else:
        stringHueOrder = None

    #palette can take a dictionary, so this should work
    # sns.kdeplot(dataToPlot, x="Acceptor/SSC", y="BDFP/SSC", hue=labelColumn, palette=colorsToUse, fill=True, alpha=0.5, legend=False, log_scale=True, ax=ax, clip=(xlims, ylims))
    sns.kdeplot(dataToPlot, x="Acceptor/SSC", y="BDFP/SSC", hue=dataToPlot[labelColumn].astype(str), palette=colorsToUse, fill=True, alpha=alpha, legend=False, log_scale=True, ax=ax, clip=(xlims, ylims), hue_order=stringHueOrder)

    defaultXLabel = "mEos3 concentration (p.d.u.)"
    defaultYLabel = "BDFP1.6:1.6 concentration (p.d.u.)"

    ax.set_xlabel(defaultXLabel)
    ax.set_ylabel(defaultYLabel)

    if title is not None:
        ax.set_title(title)
    
    if returnFig:
        return fig

def plotBDFPAcceptorContours_lines(dataToPlot, labelColumn="manualLabel", colors=plt.get_cmap("tab10").colors, ax=None, xlims=None, ylims=None, returnFig=False, hueOrder=None, linewidths=np.linspace(0.1,2,9), title=None):
    if not ax:    
        fig = plt.figure()
        ax = plt.gca()

    if "BDFP/SSC" not in dataToPlot:
        dataToPlot["BDFP/SSC"] = dataToPlot["BDFP1.6-A"] / dataToPlot["SSC 488/10-A"]

    colorsToUse = {str(label): colors[label] for label in dataToPlot[labelColumn].unique()}
    if hueOrder is not None:
        filteredHueOrder = [label for label in hueOrder if label in dataToPlot[labelColumn].unique()]
        stringHueOrder = [str(label) for label in filteredHueOrder]
    else:
        stringHueOrder = None

    #palette can take a dictionary, so this should work
    # sns.kdeplot(dataToPlot, x="Acceptor/SSC", y="BDFP/SSC", hue=labelColumn, palette=colorsToUse, fill=True, alpha=0.5, legend=False, log_scale=True, ax=ax, clip=(xlims, ylims))
    sns.kdeplot(dataToPlot, x="Acceptor/SSC", y="BDFP/SSC", hue=dataToPlot[labelColumn].astype(str), palette=colorsToUse, fill=False, linewidths=linewidths, legend=False, log_scale=True, ax=ax, clip=(xlims, ylims), hue_order=stringHueOrder)

    defaultXLabel = "mEos3 concentration (p.d.u.)"
    defaultYLabel = "BDFP1.6:1.6 concentration (p.d.u.)"

    ax.set_xlabel(defaultXLabel)
    ax.set_ylabel(defaultYLabel)

    if title is not None:
        ax.set_title(title)
    
    if returnFig:
        return fig

#Don't use this
# def readDataToDF(filename, minAmFRET=-0.2, maxAmFRET=1.0, minAmFRETPercentile=0.01, maxAmFRETPercentile = 99.99, minAcceptorPercentile=0.1, maxAcceptorPercentile=100, minLDAPercentile=0, maxLDAPercentile=100, xAxis ="log(Acceptor)"):
#     """
#     Reads an FCS file and returns a dataframe. Also computes AmFRET, logAcceptor, and logDonor/Acceptor.
#     Drops na values, filters extreme acceptor and amfret values.
#     Filtration occurs in the order: 1. AmFRET value, 2. Acceptor percentile, 3. AmFRET percentile, 4. LDA percentile
#     If xAxis is "log(Acceptor/SSC)", that column will be made in addition to "log(Acceptor)". Only those two values are allowed for xAxis.
#     Note: this function should probably be replaced at some point to allow for more axis options and more flexibility in general.
#     Also, this function should be made to filter combinatorically, not serially (there are probably better terms, but hopefully you know what I mean).
#     Defaults may be removed at some point to emphasize making the config file specific.
#     """

#     if xAxis not in ["log(Acceptor)", "log(Acceptor/SSC)"]:
#         raise NotImplementedError("xAxis can only be either 'log(Acceptor)', or 'log(Acceptor/SSC)'")

#     #metadata, data = fcsparser.parse(filename)
#     metadata, data = fcsparser.parse(filename)


#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         #np.log10 produces a lot of warnings, but NAs get removed, so we can ignore them

#         data["AmFRET"] = data["FRET-A"] / data["Acceptor-A"]
#         data["Donor/Acceptor"] = data["Donor-A"] / data["Acceptor-A"]

#         data["logDonor/Acceptor"] = np.log10(data["Donor/Acceptor"])
#         data["log(Donor/Acceptor)"] = np.log10(data["Donor/Acceptor"])

#         data["logAcceptor"] = np.log10(data["Acceptor-A"])
#         data["log(Acceptor)"] = np.log10(data["Acceptor-A"])

#         if xAxis == "log(Acceptor/SSC)":
#             if "SSC 488/10-A" not in data.columns:
#                 raise ValueError("SSC channel not found in data!")
#             else:
#                 data["Acceptor/SSC"] = data["Acceptor-A"] / data["SSC 488/10-A"]
#                 data["log(Acceptor/SSC)"] = np.log10(data["Acceptor/SSC"])

#         if len(data) < 7:
#         # I don't like that this is hard coded, but this is only meant to avoid exceptions that break loops
#         # I think this should always work?
#             return data.dropna()

#         #This is probably very inneficient because it makes tons of copies of data.
#         #generating multiple arrays of logicals, combining them together, then applying to data would be much better.
        
#         data = data.dropna()

#         data = data[data["AmFRET"] >= minAmFRET]
#         data = data[data["AmFRET"] <= maxAmFRET]

#         if xAxis == "log(Acceptor)":
#             acceptor_bounds = np.nanpercentile(data["log(Acceptor)"], (minAcceptorPercentile,maxAcceptorPercentile))
#             data = data[data["log(Acceptor)"] >= acceptor_bounds[0]]
#             data = data[data["log(Acceptor)"] <= acceptor_bounds[1]]
#         elif xAxis == "log(Acceptor/SSC)":
#             acceptor_bounds = np.nanpercentile(data["log(Acceptor/SSC)"], (minAcceptorPercentile,maxAcceptorPercentile))
#             data = data[data["log(Acceptor/SSC)"] >= acceptor_bounds[0]]
#             data = data[data["log(Acceptor/SSC)"] <= acceptor_bounds[1]]


#         AmFRET_bounds = np.nanpercentile(data["AmFRET"], (minAmFRETPercentile,maxAmFRETPercentile))
#         data = data[data["AmFRET"] >= AmFRET_bounds[0]]
#         data = data[data["AmFRET"] <= AmFRET_bounds[1]]

#         LDA_bounds = np.nanpercentile(data["logDonor/Acceptor"], (minLDAPercentile,maxLDAPercentile))
#         data = data[data["logDonor/Acceptor"] >= LDA_bounds[0]]
#         data = data[data["logDonor/Acceptor"] <= LDA_bounds[1]]

#         return data.dropna()

def readDataFromFilelist(files):
    """
    Reads a list of files into a single dataframe and labels cells based on the order in the list.
    Also calculates some useful columns
    """
    data = pd.DataFrame()
    for i, file in enumerate(files):
        _, populationData = fcsparser.parse(file)
        populationData["label"] = i
        data = pd.concat([data, populationData])

    #make both because both are useful
    data["Acceptor/SSC"] = data["Acceptor-A"] / data["SSC 488/10-A"]
    data["log(Acceptor/SSC)"] = np.log10(data["Acceptor/SSC"])

    data["BDFP/SSC"] = data["BDFP1.6-A"] / data["SSC 488/10-A"]
    data["log(BDFP/SSC)"] = np.log10(data["BDFP/SSC"])
    
    return data
       

def gateBDFP(data, title):
    """
    gates data based on either BDFP+ or bdfp-, based on the title.
    Titles (non-case sensitive) "control" or "x3912b" are bdfp-, any other is BDFP+
    """
    #these values were chosen by visual inspection
    lowBDFPCutoff = 1
    highBDFPCutoff = 2
    
    if title.lower() == "control" or title.lower() == "x3912b":
        # dataLowBDFP = data[(data["BDFP1.6-A"] / data["SSC 488/10-A"]) <= lowBDFPCutoff]
        dataLowBDFP = data[data["BDFP/SSC"] <= lowBDFPCutoff]
        return (dataLowBDFP, "bdfp-")
    else:
        # dataHighBDFP = data[(data["BDFP1.6-A"] / data["SSC 488/10-A"]) >= highBDFPCutoff]
        dataHighBDFP = data[data["BDFP/SSC"] >= highBDFPCutoff]
        return (dataHighBDFP, "BDFP+")

def labelFromPathPoints(data, labelToVerticesDict):
    """
    labels cells within specified bounds with specified label. 
    data will have a "manualLabel" column added or rewritten.
    labelToVerticesDict must have the format {integer label: [(Acceptor/SSC, AmFRET), ...]}
    log transformation of x values will be handled internally.
    """
    data["manualLabel"] = 0
    for label, labelVertices in labelToVerticesDict.items():
        logLabelVertices = [[np.log10(point[0]), point[1]] for point in labelVertices]
        path = Path(logLabelVertices)
        data.loc[path.contains_points(data[["log(Acceptor/SSC)", "AmFRET"]]),"manualLabel"] = label
    
    return data    

def dataToPlotFromFilelist_SSC(files, title):
    """
    handles the single title filelist portion of titledFiles (parameter for DAmFRETRow_files)
    files should be a 1d list of files in population label order
    """
    data = pd.DataFrame()
    for i, file in enumerate(files):
        # populationData = DAmFRETClusteringTools.readDataToDF(file, minAmFRET=-10, maxAmFRET=10, minAmFRETPercentile=0, maxAmFRETPercentile=100, minAcceptorPercentile=0, maxAcceptorPercentile=100, xAxis="log(Acceptor/SSC)")
        populationData = readDataToDF(file, minAmFRET=-10, maxAmFRET=10, minAmFRETPercentile=0, maxAmFRETPercentile=100, minAcceptorPercentile=0, maxAcceptorPercentile=100, xAxis="log(Acceptor/SSC)")
        populationData["label"] = i
        data = pd.concat([data, populationData])
        
    #these values were chosen by visual inspection
    lowBDFPCutoff = 1
    highBDFPCutoff = 2
    
    dataLowBDFP = data[(data["BDFP1.6-A"] / data["SSC 488/10-A"]) <= lowBDFPCutoff]
    dataHighBDFP = data[(data["BDFP1.6-A"] / data["SSC 488/10-A"]) >= highBDFPCutoff]

    if title.lower() == "control" or title.lower() == "x3912b":
        return (dataLowBDFP, "bdfp-")
    else:
        return (dataHighBDFP, "BDFP+")

def DAmFRETRow(titledFiles, axs, addTitles=True, rowTitle=None, rowTitleXShift=0, rowTitleSize=12, xlims=(10**0.25, 10**3), ylims=(-0.2,1)):
    """
    titledFiles should contain a list of tuples that contain a title and a list of files:
    [(<title>, [<filename for population 0>, <filename for population 1>, ...]), ...]
    """
    for i, (title, files) in enumerate(titledFiles):
        dataToPlot, _ = dataToPlotFromFilelist_SSC(files, title)
        
        if addTitles:
            plotDAmFRETDensity(dataToPlot["Acceptor/SSC"], dataToPlot["AmFRET"], logX=True, ax=axs[i], title=title, xlims=xlims, ylims=ylims)
        else:
            plotDAmFRETDensity(dataToPlot["Acceptor/SSC"], dataToPlot["AmFRET"], logX=True, ax=axs[i], xlims=xlims, ylims=ylims)

    if rowTitle is not None:
        axs[0].text(-0.25 + rowTitleXShift, 0.5, rowTitle, transform=axs[0].transAxes, ha="center", va="center", rotation_mode="anchor", rotation=90, fontsize=rowTitleSize)

def DAmFRETRowFromData(titledDataList, axs, addTitles=True, rowTitle=None, rowTitleXShift=0, rowTitleSize=12, xlims=(10**0.25, 10**3), ylims=(-0.2,1), firstColLabelsOnly=False):
    """
    titledDataList takes a list of tuples of (title, dataframe)
    """
    for i, (title, data) in enumerate(titledDataList):        
        if addTitles:
            plotDAmFRETDensity(data["Acceptor/SSC"], data["AmFRET"], logX=True, ax=axs[i], title=title, xlims=xlims, ylims=ylims)
        else:
            plotDAmFRETDensity(data["Acceptor/SSC"], data["AmFRET"], logX=True, ax=axs[i], xlims=xlims, ylims=ylims)

    if rowTitle is not None:
        axs[0].text(-0.25 + rowTitleXShift, 0.5, rowTitle, transform=axs[0].transAxes, ha="center", va="center", rotation_mode="anchor", rotation=90, fontsize=rowTitleSize)

    if firstColLabelsOnly:
        for ax in axs[1:]:
            ax.set_xlabel(None)
            ax.set_ylabel(None)

def populationRow(titledFiles, axs, perTitleBoundaries, rowTitle=None, rowTitleXShift=0, rowTitleSize=12, colors=[[120/255, 120/255, 120/255], [249/255, 29/255, 0/255], [32/255, 25/255, 250/255]], addTitles=True, labelOrder=None, xlims=(10**0.25, 10**3), ylims=(-0.2,1)):
    for i, (title, files) in enumerate(titledFiles):
        data, _ = dataToPlotFromFilelist_SSC(files, title)
        ax = axs[i]
    
        data["logTestLabel"] = 0
        
        for label, labelVertices in perTitleBoundaries[title].items():
            logLabelVertices = [[np.log10(point[0]), point[1]] for point in labelVertices]
            # print(labelVertices)
            # print(logLabelVertices)
            path = Path(logLabelVertices)
            data.loc[path.contains_points(data[["log(Acceptor/SSC)", "AmFRET"]]),"logTestLabel"] = label

        if labelOrder is not None:
            tempLabelOrder = [value for value in labelOrder if value in data["logTestLabel"].unique()]
        
        if addTitles:
            plotDAmFRETClusters(data["Acceptor/SSC"], data["AmFRET"], data["logTestLabel"], logX=True, ylims=ylims, ax=ax, title=title, colors=colors, labelOrder=tempLabelOrder, xlims=xlims)
            
        else:
            plotDAmFRETClusters(data["Acceptor/SSC"], data["AmFRET"], data["logTestLabel"], logX=True, ylims=ylims, ax=ax, colors=colors, labelOrder=tempLabelOrder, xlims=xlims)
        
        ax.set_xscale("log")
        
    if rowTitle is not None:
        axs[0].text(-0.25 + rowTitleXShift, 0.5, rowTitle, transform=axs[0].transAxes, ha="center", va="center", rotation_mode="anchor", rotation=90, fontsize=rowTitleSize)

def populationRowFromData(titledDataList, axs, labelColumn, rowTitle=None, rowTitleXShift=0, rowTitleSize=12, colors=[[120/255, 120/255, 120/255], [249/255, 29/255, 0/255], [32/255, 25/255, 250/255]], addTitles=True, labelOrder=None, xlims=(10**0.25, 10**3), ylims=(-0.2,1), firstColLabelsOnly=False, expressionSlice=None):
    """
    titledDataList takes a list of tuples of (title, dataframe)
    labelOrder is required if expressionSlice is not none, but it really should just get set based on AmFRET values
    """
    for i, (title, data) in enumerate(titledDataList): 
        tempLabelOrder = None
        if labelOrder is not None:
            tempLabelOrder = [value for value in labelOrder if value in data[labelColumn].unique()]

        if expressionSlice is None:
            addDefaultPopulationStats = True
        else:
            addDefaultPopulationStats = False
            
        if addTitles:
            plotDAmFRETClusters(data["Acceptor/SSC"], data["AmFRET"], data[labelColumn], logX=True, ylims=ylims, ax=axs[i], title=title, colors=colors, labelOrder=tempLabelOrder, xlims=xlims, addPopulationStats=addDefaultPopulationStats)
            
        else:
            plotDAmFRETClusters(data["Acceptor/SSC"], data["AmFRET"], data[labelColumn], logX=True, ylims=ylims, ax=axs[i], colors=colors, labelOrder=tempLabelOrder, xlims=xlims, addPopulationStats=addDefaultPopulationStats)

        if expressionSlice is not None:
            axs[i].vlines(expressionSlice, ylims[0], ylims[1], color=(0.6, 0.6, 0.6), linestyles="-.")
            dataWithinSlice = data[(data["Acceptor/SSC"] >= expressionSlice[0]) & (data["Acceptor/SSC"] <= expressionSlice[1])]
            uniqueLabels, labelCounts = np.unique(dataWithinSlice[labelColumn], return_counts=True)
            labelCountDict = dict(zip(uniqueLabels, labelCounts))
    
            #start positions - will be relative to size of axis, not data
            textX = 0.01
            textYMax = 0.98
            label = tempLabelOrder[0]
            numWithLabel = labelCountDict.get(label, 0)
            # higherText = axs[i].text(textX, textYMax, f"{labelCountDict[label]}, {labelCountDict[label] / sum(labelCountDict.values()):.2f}", va="top", ha="left", color=colors[label], transform = axs[i].transAxes)
            higherText = axs[i].text(textX, textYMax, f"{numWithLabel}, {numWithLabel / sum(labelCountDict.values()):.2f}", va="top", ha="left", color=colors[label], transform = axs[i].transAxes)
            
            #make label for non-highest pop(s)
            for label in tempLabelOrder[1:]:
                numWithLabel = labelCountDict.get(label, 0)
                # higherText = axs[i].annotate(f"{labelCountDict[label]}, {labelCountDict[label] / sum(labelCountDict.values()):.2f}", xycoords=higherText, xy=(0,-1), color=colors[label], horizontalalignment="left", transform = axs[i].transAxes)
                higherText = axs[i].annotate(f"{numWithLabel}, {numWithLabel / sum(labelCountDict.values()):.2f}", xycoords=higherText, xy=(0,-1), color=colors[label], horizontalalignment="left", transform = axs[i].transAxes)
                
    
    if rowTitle is not None:
        axs[0].text(-0.25 + rowTitleXShift, 0.5, rowTitle, transform=axs[0].transAxes, ha="center", va="center", rotation_mode="anchor", rotation=90, fontsize=rowTitleSize)

    if firstColLabelsOnly:
        for ax in axs[1:]:
            ax.set_xlabel(None)
            ax.set_ylabel(None)

def populationRowWithDensityContour(titledFiles, axs, perTitleBoundaries, rowTitle=None, rowTitleXShift=0, rowTitleSize=12, colors=[[120/255, 120/255, 120/255], [249/255, 29/255, 0/255], [32/255, 25/255, 250/255]], addTitles=True, labelOrder=None, xlims=(10**0.25, 10**3), ylims=(-0.2,1)):
    for i, (title, files) in enumerate(titledFiles):
        data, _ = dataToPlotFromFilelist_SSC(files, title)
        ax = axs[i]
    
        data["logTestLabel"] = 0
        
        for label, labelVertices in perTitleBoundaries[title].items():
            logLabelVertices = [[np.log10(point[0]), point[1]] for point in labelVertices]
            # print(labelVertices)
            # print(logLabelVertices)
            path = Path(logLabelVertices)
            data.loc[path.contains_points(data[["log(Acceptor/SSC)", "AmFRET"]]),"logTestLabel"] = label

        if labelOrder is not None:
            tempLabelOrder = [value for value in labelOrder if value in data["logTestLabel"].unique()]
        
        if addTitles:
            plotDAmFRETClusters(data["Acceptor/SSC"], data["AmFRET"], data["logTestLabel"], logX=True, ylims=ylims, ax=ax, title=title, colors=colors, labelOrder=tempLabelOrder, xlims=xlims)
            
        else:
            plotDAmFRETClusters(data["Acceptor/SSC"], data["AmFRET"], data["logTestLabel"], logX=True, ylims=ylims, ax=ax, colors=colors, labelOrder=tempLabelOrder, xlims=xlims)
        
        ax.set_xscale("log")

        sns.kdeplot(data, x="Acceptor/SSC", y="AmFRET", log_scale=(True, False), clip=[(np.log10(xlims[0]),np.log10(xlims[1])), ylims], ax=ax, legend=False, alpha=0.5, linewidths=0.5, color="#000000")
        
    if rowTitle is not None:
        axs[0].text(-0.25 + rowTitleXShift, 0.5, rowTitle, transform=axs[0].transAxes, ha="center", va="center", rotation_mode="anchor", rotation=90, fontsize=rowTitleSize)

def populationRowWithDensityContourFromData(titledDataList, axs, labelColumn, rowTitle=None, rowTitleXShift=0, rowTitleSize=12, colors=[[120/255, 120/255, 120/255], [249/255, 29/255, 0/255], [32/255, 25/255, 250/255]], addTitles=True, labelOrder=None, xlims=(10**0.25, 10**3), ylims=(-0.2,1), firstColLabelsOnly=False, expressionSlice=None):
    """
    titledDataList takes a list of tuples of (title, dataframe)
    """
# populationRowFromData(titledDataList, axs, labelColumn, rowTitle=None, rowTitleXShift=0, rowTitleSize=12, colors=[[120/255, 120/255, 120/255], [249/255, 29/255, 0/255], [32/255, 25/255, 250/255]], addTitles=True, labelOrder=None, xlims=(10**0.25, 10**3), ylims=(-0.2,1), firstColLabelsOnly=False, expressionSlice=None):
    
    populationRowFromData(titledDataList, axs, labelColumn, rowTitle, rowTitleXShift, rowTitleSize, colors, addTitles, labelOrder, xlims, ylims, firstColLabelsOnly, expressionSlice)
    for i, (title, data) in enumerate(titledDataList):
        sns.kdeplot(data, x="Acceptor/SSC", y="AmFRET", log_scale=(True, False), clip=[(np.log10(xlims[0]),np.log10(xlims[1])), ylims], ax=axs[i], legend=False, alpha=0.5, linewidths=0.5, color="#000000")

    #kdeplot adds labels to everything, so remove those
    if firstColLabelsOnly:
        for ax in axs[1:]:
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        
    #     tempLabelOrder = None
    #     if labelOrder is not None:
    #         tempLabelOrder = [value for value in labelOrder if value in data["logTestLabel"].unique()]

    #     if addTitles:
    #         plotDAmFRETClusters(data["Acceptor/SSC"], data["AmFRET"], data[labelColumn], logX=True, ylims=ylims, ax=axs[i], title=title, colors=colors, labelOrder=tempLabelOrder, xlims=xlims)
            
    #     else:
    #         plotDAmFRETClusters(data["Acceptor/SSC"], data["AmFRET"], data[labelColumn], logX=True, ylims=ylims, ax=axs[i], colors=colors, labelOrder=tempLabelOrder, xlims=xlims)
    
    
    # if rowTitle is not None:
    #     axs[0].text(-0.25 + rowTitleXShift, 0.5, rowTitle, transform=axs[0].transAxes, ha="center", va="center", rotation_mode="anchor", rotation=90, fontsize=rowTitleSize)

# def plotBDFPAcceptorContours_test(dataToPlot, labelColumn="manualLabel", colors=plt.get_cmap("tab10").colors, ax=None, xlims=None, ylims=None, returnFig=False, hueOrder=None, alpha=np.linspace(0.2,0.6,9)):

def BDFPAcceptorRowFromData(titledDataList, axs, labelColumn="manualLabel", addTitles=True, rowTitle=None, rowTitleXShift=0, rowTitleSize=12, colors=[[120/255, 120/255, 120/255], [249/255, 29/255, 0/255], [32/255, 25/255, 250/255]], hueOrder=None, alpha=np.linspace(0.2,0.6,9), linewidths=np.linspace(0.1, 2, 9), xlims=None, ylims=None, firstColLabelsOnly=False, emptyFirstCol=False):
    """
    Makes a row of plots using plotBDFPAcceptorContours_test, most keywords are passed through.
    Use xlims and ylims carefully, they apply to all plots in the row regardless of BDFP+/bdfp-. 
    The defaults *should* be reasonable (for data in this run).
    Case insensitive titles of "control" or "x3912b" are assumed to be bdfp-
    alpha and linewidths are intended to be mutually exclusive
    """
    
    expressionLims_kde = (0.25, 3)
    # BDFPPositiveBDFPSSCLims_kde = (-1, 3)
    BDFPPositiveBDFPSSCLims_kde = (0, 3)
    BDFPNegativeBDFPSSCLims_kde = (-4, 1)

    #alternate labels are not allowed right now.
    defaultXLabel = "mEos3 concentration (p.d.u.)"
    defaultYLabel = "BDFP1.6:1.6 concentration (p.d.u.)"
    
    for i, (title, data) in enumerate(titledDataList):
        if emptyFirstCol and i == 0:
            #this assumes that all other columns will be BDFP+
            #this also gets the real limit values, not the log of the limit values that kdeplot uses.
            if ylims is None:
                plotYlims = (10**BDFPPositiveBDFPSSCLims_kde[0], 10**BDFPPositiveBDFPSSCLims_kde[1])
            else:
                plotYlims = (10**ylims[0], 10**ylims[1])
                
            if xlims is None:
                plotXlims = (10**expressionLims_kde[0], 10**expressionLims_kde[1])
            else:
                plotXlims = (10**xlims[0], 10**xlims[1])
                
            axs[i].set_xlim(plotXlims)
            axs[i].set_ylim(plotYlims)
            axs[i].set_xscale("log")
            axs[i].set_yscale("log")
            axs[i].set_xlabel(defaultXLabel)
            axs[i].set_ylabel(defaultYLabel)
            continue
                
        
        if ylims is None:
            if title.lower() == "control" or title.lower() == "x3912b":
                plotYlims_kde = BDFPNegativeBDFPSSCLims_kde
            else:
                plotYlims_kde = BDFPPositiveBDFPSSCLims_kde
        else:
            plotYlims_kde = ylims
        if xlims is None:
            plotXlims_kde = expressionLims_kde
        else:
            plotXlims_kde = xlims

        if linewidths is not None:
            if addTitles:
                plotBDFPAcceptorContours_lines(data, ax=axs[i], labelColumn=labelColumn, colors=colors, xlims=plotXlims_kde, ylims=plotYlims_kde, hueOrder=hueOrder, linewidths=linewidths, title=title)
            else:
                plotBDFPAcceptorContours_test(data, ax=axs[i], labelColumn=labelColumn, colors=colors, xlims=plotXlims_kde, ylims=plotYlims_kde, hueOrder=hueOrder, linewidths=linewidths)

            
        else: # use filled version by default
            if addTitles:
                plotBDFPAcceptorContours_test(data, ax=axs[i], labelColumn=labelColumn, colors=colors, xlims=plotXlims_kde, ylims=plotYlims_kde, hueOrder=hueOrder, alpha=alpha, title=title)
            else:
                plotBDFPAcceptorContours_test(data, ax=axs[i], labelColumn=labelColumn, colors=colors, xlims=plotXlims_kde, ylims=plotYlims_kde, hueOrder=hueOrder, alpha=alpha)
            
        #kdeplot automatically sets the limits to fit the data, not based on what was clipped
        plotXlims = (10**plotXlims_kde[0], 10**plotXlims_kde[1])
        plotYlims = (10**plotYlims_kde[0], 10**plotYlims_kde[1])
        axs[i].set_xlim(plotXlims)
        axs[i].set_ylim(plotYlims)
        axs[i].set_xscale("log")
        axs[i].set_yscale("log")
        
    if rowTitle is not None:
        axs[0].text(-0.25 + rowTitleXShift, 0.5, rowTitle, transform=axs[0].transAxes, ha="center", va="center", rotation_mode="anchor", rotation=90, fontsize=rowTitleSize)

    if firstColLabelsOnly:
        for ax in axs[1:]:
            ax.set_xlabel(None)
            ax.set_ylabel(None)
