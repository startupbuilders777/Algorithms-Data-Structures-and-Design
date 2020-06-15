#use "ListOperations.ml"

(* Recursive Selection Sort is O(n^2) *)

let rec selectionSort lst = 
    let getFirst (a,_) = a in
    let getSecond (_, a) = a in 
    match lst with
        | [] -> []
        | _ :: _  -> 
        let minimum = getFirst(findMinimumWithIndex lst) in
        let minimumIndex = getSecond(findMinimumWithIndex lst) in
        minimum :: selectionSort(removeIndexedElement(lst)(minimumIndex));;
