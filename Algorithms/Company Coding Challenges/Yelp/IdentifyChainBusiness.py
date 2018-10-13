# DATASET:

'''
1. A LIST OF BUSINESS CONTAINING ONLY POPULAT CHAIns (Starbucks, Walmart, ) format: CHAIN_NAME - LOCATION - BUSINESS_ID

2. A particular location

Starbucks - Seattle - 101
Peets Coffee - San Francisco - 102
Whole Foods - Austin - 103

'''

input = [
    "Starbucks - Seattle - 101", 
    "Peets Coffee - San Francisco - 102", 
    "Whole Foods - Austin - 103",
    "Starbucks - San Francisco  - 104",
    "Peets Coffee - Austin - 105",
    "Starbucks - Austin - 106",
    "Whole Foods - Austin - 103",
    "Whole Foods - Austin - 107",
    "Austin" # last line of input indicates location. go

]
# output for 



def fnc(input): 
    
    data = {} # key is county, value is another hashmap with {store name :  (set(ids), amount) } 
    for i in range(0, len(input) - 1):
        (store, county, idx) = input[i].split(" - ")

        if(data.get(county) is not None):
            countyInfo = data.get(county)
            if(countyInfo.get(store) is not None):
                countyStoreTuple = countyInfo.get(store)
                storeIdSet =  countyStoreTuple[0]
                currAmount = countyStoreTuple[1]
                if( int(idx) not in storeIdSet):
                    currAmount += 1
                    storeIdSet.add(idx)
                    countyInfo[store] = (storeIdSet, currAmount)

            else: 
                print("fook")
                countyInfo[store] = (set( [int((idx))] ), 1)
        else: 
            data[county] = { } 
            data[county][store] = (set( [int((idx))] ), 1) 

    store_name = input[len(input) - 1]
    return data[store_name]


print(fnc(input))


# Output {'Starbucks': (set([106]), 1), 'Whole Foods': (set(['107', 103]), 2), 'Peets Coffee': (set([105]), 1)}

# YELP WANTS IN DESCENDING ORDER LIKE SO:


'''
Whole Foods - 2
Peets Coffee - 1
Starbucks - 1

'''
