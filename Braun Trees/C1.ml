type 'a brt =
  Empty
| Node of 'a * 'a brt * 'a brt

type 'a sequence = 'a brt

(* This uses a similar implementation to the merge from mergesort *)
let rec merge lst1 lst2 = match (lst1, lst2) with
   ([], ns) -> ns
 | (ms, []) -> ms
 | (n::ns, m::ms) -> n :: m :: merge ns ms

(* Algorithm description: recursively convert the left and right trees into
    lists and merge the lists as we go *)
let rec tree2list = function
  | Empty -> []
  | Node (e, left, right) -> e :: merge (tree2list left) (tree2list right)

let rec mergeTrees b1 b2 = match (b1, b2) with
  | (Empty, r) -> r
  | (l, Empty) -> l
  | () -> Node (hd, tl, getright tl)

(* This function is supposed to split a list into two parts *)
let rec break hd tl = match (hd, tl) with
  | _ -> tl 
    
let rec list2tree = function
  | [] -> Empty
  | [e] -> Node (e, Empty, Empty)
  | a :: tl -> Node (a, break (list2tree tl))
