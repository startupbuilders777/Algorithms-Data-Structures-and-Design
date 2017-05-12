let rec appendTwoLists lst1 lst2 = 
    match (lst1, lst2) with
    | (h1 :: t1, lst2) -> h1 :: append t1 lst2
    | ([], lst2) -> lst2

let rev list =
    let rec aux acc = function
      | [] -> acc
      | h::t -> aux (h::acc) t in
    aux [] list;;

let rec last lst =
    match lst with
    | [] -> None
    | [a] -> Some a
    | _ :: tl -> last tl;;

let rec at k = function
    | [] -> None
    | hd::tl -> if k = 1 then Some hd else at (k-1) tl;;

let length lst = 
    let rec length_rec counter = function
    | [] -> counter
    | hd::tl -> length_rec (counter+1) tl in
    length_rec 0 lst ;;

let findMinimumWithIndex lst =
    let rec aux min minIndex counter= function
        | [] -> (min, minIndex)
        | h :: tl when h < min -> aux h counter (counter+1) tl
        | _ :: tl -> aux min minIndex (counter + 1) tl in
    aux (List.hd lst) 0 0 lst

let removeIndexedElement lst index = 
    let rec aux index counter = function
        | [] -> []
        | hd::tl when counter = index -> tl
        | hd::tl -> hd::aux index (counter + 1) tl in
    aux index 0 lst 

let getElement lst index = 
    let rec aux index counter = function
          [] -> None
        | hd :: tl when counter = index -> Some hd
        | hd :: tl -> aux index (counter+1) tl in
    aux index 0 lst;;

let swapListElements index1 index2 lst = 
    let getValueFromOption item = 
        match item with
        | None -> 0 (*This is a bad case that should be removed*)
        | Some x -> x in 
    let rec aux index1 index2 item1 item2 counter = function
        | [] -> []
        | hd :: tl when counter = index1 -> (getValueFromOption item2) :: (aux index1 index2 item1 item2 (counter+1) tl)
        | hd :: tl when counter = index2 -> (getValueFromOption item1) :: (aux index1 index2 item1 item2 (counter+1) tl)
        | hd :: tl -> hd :: (aux index1 index2 item1 item2 (counter+1) tl) in
    aux index1 index2 (getElement lst index1) (getElement lst index2) 0 lst  
    