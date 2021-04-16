from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math


warnings.simplefilter("ignore")
sns.set()


def func1(matches):
    # calculate number decision per season
    decision_per_season = matches.groupby(matches.index.year)[
        "toss_decision"].value_counts()

    temp_arr = []

    for idx, val in decision_per_season.iteritems():
        season = idx[0]
        decision_type = idx[1]
        temp_arr.append([season, decision_type, val])

    temp_df = pd.DataFrame(data=temp_arr, columns=[
                           "Season", "TossDecision", "Count"])

    g = sns.factorplot(x='Season', y='Count',
                       hue='TossDecision', data=temp_df, kind='bar')

    g.fig.set_size_inches(13, 7)
    plt.title("Total Decisions across the seasons")
    plt.savefig('temp.eps', format='eps')
    plt.show()


def func2(matches):
    plt.subplots(figsize=(10, 6))

    ax = matches['toss_winner'].value_counts().plot.bar(
        width=0.9, color=sns.color_palette('RdYlGn', 20))
    for p in ax.patches:
        ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
    plt.title("Maximum Toss Winners")
    plt.savefig('max_toss_winner.eps', format='eps')
    plt.show()


def func3(matches):
    temp_df = matches[matches["toss_winner"] == matches["winner"]]
    slices = [temp_df.shape[0], matches.shape[0] - temp_df.shape[0]]
    print(slices)
    labels = ["Yes", "No"]
    fig, ax = plt.subplots()
    ax.pie(slices, explode=[0, 0.01], labels=labels,
           shadow=False, startangle=90, autopct='%1.1f%%')
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    plt.title("is Toss Winner Also the Match Winner?")
    plt.savefig('is_toss_winner_match_winner.eps', format='eps')
    plt.show()


def func4(matches):
    # Matched played across each season
    matches_per_year = matches.groupby(matches.index.year).count()["id"]
    year = matches_per_year.index.tolist()
    count = matches_per_year.tolist()
    temp_df = pd.DataFrame({"Season": year, "TotalMatches": count})
    g = sns.factorplot(x='Season', y='TotalMatches', data=temp_df,
                       kind='bar', palette=sns.color_palette('winter'))
    g.fig.set_size_inches(13, 7)
    plt.title("Match played across seasons")
    plt.savefig('matched_played_across_seasons.eps', format='eps')
    plt.show()


def func5(matches, deliveries):
    # Total runs across the Sesons
    temp_df = matches.merge(right=deliveries, how="left", on="id")
    temp_df = temp_df[["date", "total_runs"]]
    temp_df = temp_df.set_index(["date"])
    total_runs_per_season = temp_df.groupby(
        temp_df.index.year).sum().reset_index()
    total_runs_per_season.columns = ["Season", "TotalRuns"]
    g = sns.lineplot(data=total_runs_per_season,
                     x="Season", y="TotalRuns", markers="o")

    plt.gcf().set_size_inches(10, 6)
    plt.title("Total Runs Across the Seasons")
    plt.savefig('total_runs_across_the_seasons.eps', format='eps')
    plt.show()


def func6(matches, deliveries):
    # Average Number of runs per match across the season
    temp_df = matches.merge(right=deliveries, how="left", on="id")
    temp_df = temp_df[["date", "total_runs"]]
    temp_df = temp_df.set_index(["date"])
    matches_count_per_season = matches.groupby(
        matches.index.year).count()["id"]
    total_runs_per_season = temp_df.groupby(
        temp_df.index.year).sum().reset_index()

    total_runs_per_season.columns = ["Season", "TotalRuns"]
    # print(matches_count_per_season)
    average_runs_per_season = total_runs_per_season.copy()
    average_runs_per_season["AvgRuns"] = total_runs_per_season["TotalRuns"] / \
        np.array(matches_count_per_season.tolist())
#    print(average_runs_per_season)
    sns.lineplot(data=average_runs_per_season, x="Season", y="AvgRuns")
    plt.gcf().set_size_inches(10, 6)
    plt.title("Average runs across the Season")
    plt.savefig('average_runs_across_the_season.eps', format='eps')
    plt.show()


def func7(matches, deliveries):
    temp_df = matches.merge(right=deliveries, how="left", on="id")
    temp_df["date"] = pd.to_datetime(temp_df["date"])
    temp_df = temp_df.set_index("date")
    temp_df = temp_df.groupby(temp_df.index.year)["batsman_runs"]
    temp_arr = []

    for year, score_arr in temp_df:
        # print(key)
        freq = score_arr.value_counts()
        temp_arr.append([year, freq[4], freq[6]])

    temp_df = pd.DataFrame(data=temp_arr, columns=["Year", "Four", "Six"])
    plt.plot("Year", "Four", data=temp_df, marker="o")
    plt.plot("Year", "Six", data=temp_df, marker="x")
    plt.title("Sixes and Fours Across the Season")
    plt.savefig('six_four_across_the_season.eps', format='eps')
    plt.legend()
    plt.show()


def func8(matches, deliveries):
    runs_per_over = deliveries.pivot_table(
        index=['over'], columns='batting_team', values='total_runs', aggfunc=sum)
    # print(runs_per_over)
    # match played by teams
    #print(matches[["team1", "team2"]])
    match_played = {}
    for idx, row in matches[["team1", "team2"]].iterrows():
        if row.team1 not in match_played:
            match_played[row.team1] = 0
        if row.team2 not in match_played:
            match_played[row.team2] = 0

        match_played[row.team1] += 1
        match_played[row.team2] += 1
    matches_played_byteams = pd.DataFrame(
        match_played.items(), columns=["Team", "Total Matches"])
    matches_played_byteams = matches_played_byteams.reset_index(drop=True)
    # print(runs_per_over.head(10))
    # print(matches_played_byteams.head(10))
    teams_gt50 = matches_played_byteams[matches_played_byteams["Total Matches"] > 50]["Team"]
    teams_gt50 = teams_gt50.tolist()
    temp_df = runs_per_over[teams_gt50]
    temp_df.index = range(1, 21)
    temp_df.plot(figsize=(10, 5), color=[
                 "b", "r", "#Ffb6b2", "g", 'brown', 'y', '#6666ff', 'black', '#FFA500'])
    plt.xticks(range(1, 21))
    plt.title("Runs per over by teams across seasons")
    plt.xlabel("Over")
    plt.ylabel("Runs")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.savefig('runs_per_over_by_teams_seasons.eps', format='eps')
    plt.show()


def func9(matches):
    # Favourite Ground
    # plt.subplots(figsize=(25, 7))
    ax = matches['venue'].value_counts().sort_values(
        ascending=True).plot.barh(figsize=(50, 10), width=0.9, color=sns.color_palette("tab10"), rot=20)
    ax.set_xlabel('Grounds')
    ax.set_ylabel('count')

    plt.title("Favourite Grounds")
    plt.savefig('fav_ground.eps', format='eps')
    plt.show()


def func10(matches):
    # Top man of the match
    matches["player_of_match"].value_counts()[:10].plot.bar(
        figsize=(9, 8), color=sns.color_palette("tab10"), rot=60)
    plt.title("Top 10 Man of the Match")
    plt.xlabel("Player")
    plt.ylabel("Count")
    plt.savefig('top_10_man_of_the_match.eps', format='eps')
    plt.show()


def func11(deliveries):
    # Top 10 batsman
    plt.subplots(figsize=(10, 6))
    max_runs = deliveries.groupby(['batsman'])['batsman_runs'].sum()
    ax = max_runs.sort_values(ascending=False)[:10].plot.bar(
        width=0.8, rot=25, color=sns.color_palette("tab10"))
    for p in ax.patches:
        ax.annotate(format(p.get_height()), (p.get_x() +
                                             0.1, p.get_height()+50), fontsize=15)
    plt.title("Top 10 batsman")
    plt.savefig('top_10_batsman.eps', format='eps')
    plt.show()


def func12(deliveries):
    # Top batsman who hit maximum number of 6 and 4
    temp_df = deliveries.groupby("batsman")
    batsman_4_6_count = {}
    for name, df in temp_df:
        if not (name in batsman_4_6_count):
            batsman_4_6_count[name] = {}
        # print(name)
        batsman_4_6_count[name]["6"] = df[df["batsman_runs"] == 6].shape[0]

        batsman_4_6_count[name]["4"] = df[df["batsman_runs"] == 4].shape[0]
    # print(batsman_4_6_count)
    temp_df = pd.DataFrame(batsman_4_6_count).T.sort_values(by=["6", "4"])
    temp_df.tail(10).sort_values(by=["4", "6"]).plot.bar(
        width=0.9, rot=25, color=sns.color_palette("tab10"), figsize=(10, 7))
    plt.title("Top batsman with number of 4's and 6's")
    plt.savefig('top_batsman_4_6.eps', format='eps')
    plt.show()


def func13(deleveries):
    # Top scorer
    temp_df = deleveries.groupby("batsman").sum(
    )["total_runs"].sort_values(ascending=False)
    # print(temp_df.head(10))
    # temp_df.plot.scatter()
    # plt.subplots(figsize=(12, 5))

    df_for_plot = pd.DataFrame({"BatsMan": temp_df.head(
        10).index, "Runs": temp_df.head(10).values})
    plt.figure(figsize=(12, 5))
    g = sns.barplot(data=df_for_plot, x="BatsMan", y="Runs")
    plt.title("Top 10 Batsman")
    #g.set_size_inches(13, 7)
    plt.savefig('top_10_batsman.eps', format='eps')
    plt.show()


def func14(deleveries):
    # top individual scores
    top_scorer = deleveries.groupby(["id", "batsman", "batting_team"])[
        "batsman_runs"].sum().reset_index().sort_values(ascending=False, by="batsman_runs")
    print(top_scorer.head(10))


def func15(delivery):

    top_scores = delivery.groupby(["id", "batsman", "batting_team"])[
        "batsman_runs"].sum().reset_index().sort_values(ascending=False, by="batsman_runs")
    swarm = ['CH Gayle', 'V Kohli', 'G Gambhir', 'SK Raina',
             'YK Pathan', 'MS Dhoni', 'AB de Villiers', 'DA Warner']
    scores = delivery.groupby(["id", "batsman", "batting_team"])[
        "batsman_runs"].sum().reset_index()
    scores = scores[top_scores['batsman'].isin(swarm)]
    sns.swarmplot(x='batsman', y='batsman_runs', data=scores,
                  hue='batting_team', palette='Set1')
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    # plt.ylim(-10,200)
    plt.title("Individual Scores By Top Batsman each Inning")
    plt.savefig('score_by_top_batsman.eps', format='eps')
    plt.show()


def func16(deliveries):
    # Runs scored by batsman across seasons
    print(deliveries.groupby("batsman").sum()[
          "batsman_runs"].sort_values(ascending=False)[:20])


def main():
    matches = pd.read_csv("./data/IPL Matches 2008-2020.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.set_index("date", drop=False)
    delivery = pd.read_csv("./data/IPL Ball-by-Ball 2008-2020.csv")

    matches.replace(["Delhi Capitals"], ["Delhi Daredevils"], inplace=True)
    matches.replace(['Mumbai Indians', 'Kolkata Knight Riders', 'Royal Challengers Bangalore', 'Deccan Chargers', 'Chennai Super Kings',
                     'Rajasthan Royals', 'Delhi Daredevils', 'Gujarat Lions', 'Kings XI Punjab',
                     'Sunrisers Hyderabad', 'Rising Pune Supergiants', 'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiant'], ['MI', 'KKR', 'RCB', 'DC', 'CSK', 'RR', 'DD', 'GL', 'KXIP', 'SRH', 'RPS', 'KTK', 'PW', 'RPS'], inplace=True)

    delivery.replace(['Mumbai Indians', 'Kolkata Knight Riders', 'Royal Challengers Bangalore', 'Deccan Chargers', 'Chennai Super Kings',
                      'Rajasthan Royals', 'Delhi Daredevils', 'Gujarat Lions', 'Kings XI Punjab',
                      'Sunrisers Hyderabad', 'Rising Pune Supergiants', 'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiant'], ['MI', 'KKR', 'RCB', 'DC', 'CSK', 'RR', 'DD', 'GL', 'KXIP', 'SRH', 'RPS', 'KTK', 'PW', 'RPS'], inplace=True)

    # Data Cleaning and Pre-Processing
    matches = matches.drop(
        ["umpire1", "umpire2", "method", "result_margin", "eliminator"], axis=1)

    city_mode = matches["city"].mode()
    # replacing "nan" with mode of city column
    for idx, city in matches["city"].items():
        if str(city) == "nan":
            matches.loc[idx, "city"] = city_mode.loc[0]

    for idx, player in matches["player_of_match"].items():
        if str(player) == "nan":
            matches["player_of_match"][idx] = matches["player_of_match"].mode().loc[0]

    # for winner column replacing with is not appropriate
    # we are just choosing random team
    for idx, team in matches["winner"].items():
        if str(team) == "nan":
            choice_arr = [matches["team1"][idx], matches["team2"][idx]]
            matches["winner"][idx] = choice_arr[math.floor(
                np.random.rand() * len(choice_arr))]

    print("Total matches played from 2008 to 2020: ", matches.shape[0], "\n")
    print("Venues played at: ", matches["city"].unique(), "\n")
    print("Teams: ", matches["team1"].unique(), "\n")
    print("Total venues played at: ", matches["city"].nunique(), "\n")

    print(matches["player_of_match"].value_counts().idxmax(),
          " has most man of the matches award!", "\n")

    print(matches["winner"].value_counts().idxmax(),
          " has the highest number of match wins!", "\n")

    # print(matches["toss_winner"].value_counts().index.tolist())
    # print(matches["toss_winner"].value_counts().tolist())

    # func1(matches)
    # func2(matches)
    # func3(matches)
    # func4(matches)
    # func5(matches, delivery)
    # func6(matches, delivery)
    # func7(matches, delivery)
    # func8(matches, delivery)
    # func9(matches)
    # func10(matches)
    # func11(delivery)
    # func12(delivery)
    # func13(delivery)
    # func14(delivery)
    # func15(delivery)
    func16(delivery)


if __name__ == "__main__":
    main()
