#use "ListOperations.ml"

(* [[1;2];[3];[4;5;6];[5;7]] => [1;2;3;4;5;6;5;7]*)
let appendLists lst = 
    let rec aux = function 
        | [] -> []
        | [a] -> a
        | []::rest -> aux rest
        | (h::t)::rest -> h :: aux (t::rest) in
    aux lst;;
      

let rec quicksort lst =
    let rec aux left right pivot = function
        | [] ->  appendLists([quicksort(left); [pivot]; quicksort(right)])
        | hd :: tl 
            -> match hd < pivot with
                | true -> aux (hd::left) right pivot tl 
                | false -> aux left (hd::right) pivot tl in
    match lst with 
    | [] -> []
    | (hd::tl as list) ->
        let pivot = (List.hd list) in
        let restOfList = (List.tl list) in 
        aux [] [] pivot restOfList;;
        

