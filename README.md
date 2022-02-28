# TSH_Control: A diary for your blood parameters

TSH_Control is a health dashboard to monitor health parameters. It is coded using `Dash`.

I use it to track my own TSH levels and medicament dosage, as it needs to be monitored regularly.

The dashboard is publicly accessible from this [URL](https://4emduf.deta.dev/).

![Alt text](./data/other/screenshot.jpg?raw=true "Title")

# It features:

- a [Deta app](https://www.deta.sh/) located in TSH_Control/TSH_Control and deployed to a cloud server.

- an XML parser to analyze Apple Health data. 
 
- a preprocessing pipeline (`preprocess.py`), which goes through the raw data files and populates the clean folder. 
  Preprocessing occurs locally as Pandas is too big to run on Deta's free tier.
 
- As Apple Health doesn't support yet mood ratings, i am using Daylio app for this. The data that is plotted here is a manual .csv export.

- Blood tests and dosage data could probably be stored in Apple Health, but I keep track of them in a csv file.

- A correlation matrix to understand the data and relationships between datasets.

# Installation

Run `make setup.env` to setup the Python and the deta environments. Python environment is needed to preprocess data and locally run Dash app. Deta is needed to deploy the app and datasets to Deta cloud. You need a `config.txt` to contain Deta credentials with `projectKey` and `projectID` keys.
