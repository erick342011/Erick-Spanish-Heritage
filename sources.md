# Sources and indicator definitions

## World Bank API

The application retrieves annual observations from the World Bank Indicators API:

`https://api.worldbank.org/v2/country/{ISO3}/indicator/{INDICATOR_CODE}?format=json&date={START}:{END}`

API documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation

Country mappings used:
- Mexico (MX) → `MEX`
- Brazil (BR) → `BRA` — included as a Latin American comparison country
- Argentina (AR) → `ARG`
- Colombia (CO) → `COL`
- Chile (CL) → `CHL`
- Peru (PE) → `PER`
- Venezuela (VE) → `VEN`
- Ecuador (EC) → `ECU`
- Uruguay (UY) → `URY`

## Indicator mappings

| Indicator | World Bank code | Definition / unit |
|---|---|---|
| Life expectancy at birth | `SP.DYN.LE00.IN` | Expected years of life at birth under current mortality patterns; years |
| Unemployment, national estimate | `SL.UEM.TOTL.NE.ZS` | Unemployed share of the labor force; percent |
| GDP per capita | `NY.GDP.PCAP.CD` | GDP divided by midyear population; current US dollars per person. Not average wealth or household income. |
| Crude birth rate | `SP.DYN.CBRT.IN` | Annual live births per 1,000 people; not a total birth count |
| Net migration | `SM.POP.NETM` | Net migration measure; not an emigration count |
| Intentional homicide rate | `VC.IHR.PSRC.P5` | Intentional homicides per 100,000 people |
| Consumer price inflation | `FP.CPI.TOTL.ZG` | Annual percentage change in consumer prices |
| Central government debt | `GC.DOD.TOTL.GD.ZS` | Central government debt as a share of GDP; percent |
| Total reserves including gold | `FI.RES.TOTL.CD` | Total reserves including gold; current US dollars |
| External balance on goods and services | `NE.RSB.GNFS.ZS` | External balance on goods and services as a share of GDP; percent |

## Retrieval and data handling

The requested time window is current year minus 70 through current year minus 1, inclusive. Actual available years vary by country and indicator. Missing values are omitted from the working observations and are not replaced with synthetic values. The app records the retrieval date in the report.

## Mathematical method

Polynomial regression uses centered time `t = year - base year`. In-sample R-squared and mean absolute error (MAE) summarize fit on the same observations used to estimate the model. They are not validated prediction intervals. A chronological holdout MAE is displayed only when enough observations exist. Extrapolations can be implausible, especially for high-degree polynomials.

The separate function lab uses SymPy for symbolic derivatives and NumPy for numerical plotting. Numerical critical-point and inflection-point classifications are approximate; candidate roots are classified only when derivative sign tests support the classification. Difficult roots and discontinuities may require manual review.
