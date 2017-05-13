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
        

(*merge Elements 
[[a,b], e, [c,d], [f,g], [[[e,f], g],h],i] => [a,b,e,c,d,f,g,e,f,g,h,i] 

This wont work because you need a nested list structure whihc OCAML does not define.
let merge nodes = 
    let rec append lst1 lst2 = 
        match (lst1, lst2) with
            | ([], []) -> []
            | ([], lst2) -> lst2
            | (h1::t1, lst2) -> h1 :: append t1 lst2 in
    let rec aux output = function
        | [] -> output
        | hd :: tl 
            -> match hd with 
                | (hdh :: hdt as innerList) -> aux (append (aux [] innerList) output) tl
                | _ -> aux (hd::output) tl in
    aux [] nodes
*)

