# TSH_Control: A diary for your blood parameters

TSH_Control is a health dashboard to monitor health parameters. 

I use it to track my own TSH levels and medicament dosage, as it needs to be monitored regularly.

The dashboard is publicly accessible from this [URL](https://4emduf.deta.dev/).

Most of the preprocessing occurs locally.


# It features:

- a [Deta app](https://www.deta.sh/) located in TSH_Control/TSH_Control and deployed to a cloud server.

- an XML parser to analyze Apple Health data. 
 
- a preprocessing pipeline (`preprocess.py`), which goes through the raw data files and populates the clean folder.
 
- As Apple Health doesn't support yet mood ratings, i am using Daylio app for this. The data that is plotted here is a manual .csv export.

- Blood tests and dosage data could probably be stored in Apple Health, but I keep track of them in a csv file.

