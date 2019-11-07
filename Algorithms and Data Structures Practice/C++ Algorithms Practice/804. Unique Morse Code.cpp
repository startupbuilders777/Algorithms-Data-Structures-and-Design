/*

International Morse Code defines a standard encoding where each letter is mapped to a series of dots and dashes, as follows: "a" maps to ".-", "b" maps to "-...", "c" maps to "-.-.", and so on.

For convenience, the full table for the 26 letters of the English alphabet is given below:

[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
Now, given a list of words, each word can be written as a concatenation of the Morse code of each letter. For example, "cba" can be written as "-.-..--...", (which is the concatenation "-.-." + "-..." + ".-"). We'll call such a concatenation, the transformation of a word.

Return the number of different transformations among all words we have.

Example:
Input: words = ["gin", "zen", "gig", "msg"]
Output: 2
Explanation: 
The transformation of each word is:
"gin" -> "--...-."
"zen" -> "--...-."
"gig" -> "--...--."
"msg" -> "--...--."

There are 2 different transformations, "--...-." and "--...--.".

*/


 #include <unordered_set>
    #include <vector>
    #include <string>
    #include <iostream>

    using namespace std;
    

class Solution {
    
    
   
public:
    int uniqueMorseRepresentations(vector<string>& words) {
        
        auto codes = vector<string>
        {".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."};
        
        
        string morse = "";
        unordered_set<string> s; 
        
            
        for(string word: words) {
            for (char c: word) {
                cout << (c-'a') << " and " << c << endl;
                    
                    
                string piece = codes[c - 'a'];
                // cout << piece << endl;
                
                morse += piece;
            }
        
            s.insert(morse);
            morse = "";
                    
                
            }
                    
    for (auto _s: s) {
        cout << _s << endl;
    }
    
    return s.size();
    }
    
    
};


//other solutions:

//Faster:

class Solution {
public:
    int uniqueMorseRepresentations(vector<string>& words) {
        string mapping[] = {".-", "-...", "-.-.", "-..", ".", "..-.",
                         "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---",
                         ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-",
                         "-.--", "--.."};
        set<string> myset;
        for (auto &i : words) {
            string temp = "";
            for (auto &j : i) {
                temp += mapping[j - 'a'];
            }
            myset.insert(temp);
        }
        return myset.size();
    }
};

