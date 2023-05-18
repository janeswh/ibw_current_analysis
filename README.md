## About
This repo contains the data analylsis pipeline used to process the electrophysiological data obtained from recordings in mouse olfactory bulb slices. The project asks the question:
> **Do postnatal-born granule cells generated at different timepoints in the animal's life differ in their functional connectivity with primary excitatory cells in the olfactory bulb?**

Below is a brief overview of the analysis steps performed. Experimental results are not shown as the project is yet unpublished.

## Data collection
Electrophysiological recordings were obtained using custom-written software using [Igor 8](https://www.wavemetrics.com/) from mitral cells (MC) and tufted cells (TC) in the olfactory bulb during optogenetic stimulation of olfactory granule cells (GC).

## Data analysis

### Identifying inhibitory postsynaptic currents (IPSCs) in MCs and TCs
Using the [MOD](https://pub.ista.ac.at/~schloegl/software/mod/rc4/README.html) script in MATLAB, light-evoked IPSCs were automatically detected in all recorded cells, along with their peaks and kinetics.

![](https://github.com/janeswh/ibw_current_analysis/blob/main/figs/single_annotated.png)

### Quantifying the frequency of light-evoked IPSCs
The number of light-evoked IPSCs in every trial were summed to create a peristimulus time histogram denoting the frequency of inhibitory events. The mean current amplitude of all the trials was also calculated.

![](https://github.com/janeswh/ibw_current_analysis/blob/main/figs/single_freq.png)

### Determination of connectivity
To determine whether a cell showed light-evoked inhibitory response to optogenetic activation of GCs, the average IPSC frequency with and without light stimulation was compared using paired t-tests and Kolmogorov-Smirnov test.

![](https://github.com/janeswh/ibw_current_analysis/blob/main/figs/single_stats.png)

### Comparison between MCs and TCs
Event kinetics were then pooled and compared between MCs and TCs following light-activation of either early-born GCs or late-born GCs. Differences in connectivity and IPSC amplitudes would suggest differences in connectivity between GCs born at different times and MCs vs. TCs.

## Credits

* [MOD detection](https://pub.ista.ac.at/~schloegl/software/mod/rc4/README.html) code by Alois Schlögl