import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import seaborn as sns

cbb = pd.read_csv(r"/Users/alexcasieri/PycharmProjects/hello-world/Data/cbb.csv")
print(cbb.head())
print()

#explanation
print("I added the winning percentage column in order to compare different teams to see "
      "which was better than which.")
wins = cbb["W"]
total_games = cbb["G"]
cbb["Win%"] = (wins/total_games).round(decimals = 4)
print(cbb.head())
print()

#explanation for value counts
print("Using the value counts showed me that there were 351 teams across 5 years.")
print(cbb["YEAR"].value_counts())
print()

#table to show evidence that effective field goal percetage is important
print("I made a function called 'nonzero' that creates a list with boolean values True or False to represent whether "
      "a value is above a certain cutoff. I then used this function to input the list of teams and their winning "
      "percentages, while the cutoff  was .5. By doing this I was abe t extract the winnning teams (won more than half "
      "of their games) from the losing teams from the whole dataset. I chose to use the groupby function"
      "to help with this. It is clear from the mini-table that teams that won more than half of their games average"
      "approximately 3 percentage points higher of an effective field goal percentage.")
def nonzero(list_of_numbers, cutoff):
    list1 = []
    for i in list_of_numbers:
        if i > cutoff:
            list1.append(True)
        else:
            list1.append(False)
    return list1
new_col = nonzero(cbb["Win%"], .5)
cbb["ABOVE 500"] = new_col
data = cbb.groupby("ABOVE 500")["EFG_O"].mean()
print(data)
print()

#scatter plot of the two variables
print("This is a scatterplot of effective field goal percentage vs winning percetage. "
      "There is a clear positive correlation between these two variables.")
EFGO_vs_win = plt.scatter(cbb["EFG_O"], cbb["Win%"]*100, alpha=.5)
plt.xlabel("Effective Field Goal Percentage Shot")
plt.ylabel("Winning Percentage")
plt.show()
print()

data2 = cbb.groupby("CONF")["Win%"].mean().sort_values(ascending = False)
median_conf_win = data2.median()
max_conf_win = data2.max()
if (data2.min() == .37857000000000013):
    min_conf_win = .37857

unique = []
for i in cbb["CONF"]:
    if i not in unique:
        unique += [i]
for i in unique:
    conference = cbb[cbb["CONF"]==i]
    if conference["Win%"].mean() == max_conf_win:
        max_conf = i

for i in unique:
    conference = cbb[cbb["CONF"]==i]
    if conference["Win%"].mean() == min_conf_win:
        min_conf = i


for i in unique:
    conference = cbb[cbb["CONF"]==i]
    if conference["Win%"].mean() == median_conf_win:
        median_conf = i


years = []

for i in cbb["YEAR"]:
    if i not in years:
        years += [i]
        years.sort()
list_of_EFGO_for_max = []
list_of_EFGO_for_min = []
list_of_EFGO_for_median = []

for i in years:
    specific_year = cbb[cbb["YEAR"] == i]

    conference_in_year_max = specific_year[specific_year["CONF"]== max_conf]
    EFGOmax = conference_in_year_max["EFG_O"]
    list_of_EFGO_for_max += [format(EFGOmax.mean(), ".5f")]

    conference_in_year_min = specific_year[specific_year["CONF"]== min_conf]
    EFGOmin = conference_in_year_min["EFG_O"]
    list_of_EFGO_for_min += [format(EFGOmin.mean(), ".5f")]

    conference_in_year_median = specific_year[specific_year["CONF"] == median_conf]
    EFGOmed = conference_in_year_median["EFG_O"]
    list_of_EFGO_for_median += [format(EFGOmed.mean(), ".5f")]

plt.figure(figsize = (10,6))
barwidth = .2
index = np.arange(5)
plt.bar(index - barwidth, list_of_EFGO_for_max, width = barwidth, color = "g", label = max_conf + " - highest win%")
plt.bar(index, list_of_EFGO_for_median, width = barwidth, color = "r", label = median_conf + " - median win%")
plt.bar(index + barwidth, list_of_EFGO_for_min, width = barwidth, color = "b", label = min_conf + " - lowest win%")
plt.legend()
plt.xticks(index, years)
plt.xlabel("Year")
plt.ylabel("Average Effective Field Goal % Shot")
plt.title("Effective Field Goal Percentage Shot by Conferences from 2015 - 2019")

ax = plt.gca()
ax.set_ylim([35,60])
plt.show()

print("First, I used the groupby function to list winning percentages by conference over the 5 year span. Then I "
      "found the conference with the overall highest winning percentage, the lowest winninng percentage, and the "
      "median winning percentage of the dataset. I then found the average effective field goal percentage for each "
      "of these conferences in each year. I did this to find out if there was a link between effective field goal "
      "percentage and winning percentage by conference. As shown by the bar graph, the conference with the highest "
      "average winninng percentage usually had the highest effective field goal percentage as well. The conference "
      "with the lowest average winning percentage usually had the lowest effective field goal percentage as well. "
      "There seems to be a of link between effective field goal percentage shot and winning percentage.")
print()

efg = cbb["EFG_O"]
win = cbb["Win%"]
correlation = cbb[["EFG_O", "Win%"]].corr()
r_value = correlation.iloc[0,1]
print("Calcualted R value =" , format(r_value, ".5f"))
print()

print("Statistical Summary:")
winning_p = cbb["Win%"]*100
effective_fg = cbb["EFG_O"]
statistics = sm.ols(formula = 'winning_p ~ effective_fg', data = cbb).fit()
print(statistics.summary())
print(statistics.params)
value = statistics.params["effective_fg"]
print()
print(f"For every increase in percentage point of effective field goal p ercentage shot, a given team has a {value} "
      f"increase in win percentage.")
print(statistics.pvalues)
print()
sns.regplot(x = "EFG_O", y = "Win%", data = cbb)
plt.show()
sns.regplot(x = "EFG_O", y = statistics.resid, data = cbb)
plt.show()
print(statistics.resid.head())
