import pandas as pd
import json
from _collections import defaultdict
df = pd.read_csv('topical_chat.csv')

tags = list(df.sentiment.unique())


# a dictionary which has the key as the sentiment and value as the list of words in that sentiment

hmap = defaultdict(list)


for i in range(len(df)):
    hmap[df["sentiment"][i]].append(df["message"][i])
    # print(hmap)

responses = [["yes, I can tell you more about it in future", "pretty interesting right!", "If you want to learn more you'll have to ask me later"],
             ["Amazing!"],
             ["hmmm"],
             ["yes, don't be surprised you'll just have to deal with it"],
             ["WHAT! why are you making that face you"],
             ["issok, you don't have to be sad anymore"],
             ["😎 why are afraid"],
             ['calm down', 'ayyy chilll!!!', 'I better call saul to settle your anger']]

intents = {"intents" : []}
i = 0
print(hmap.keys())
for key in hmap.keys():
    toAppend = {"tag": "", "pattern": [], "response":[]}
    toAppend["tag"] = key
    toAppend["pattern"] = hmap[key][:50]
    toAppend["response"] = responses[i]
    intents['intents'].append(toAppend)
    i += 1

with open("intents.json", "w") as outfile:
    json.dump(intents, outfile)
