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
(* Time efficiency: O(n*2)?? *)
let rec tree2list = function
  | Empty -> []
  | Node (e, left, right) -> e :: merge (tree2list left) (tree2list right)

(* The following is a naive implementation of list-to-tree *)
(* Time efficiency: O(n*log n) *)
let rec extend e = function
  | Empty -> Node (e, empty, empty)
  | Node (elem, left, right) -> Node (e, extend elem right, left)

let rec list2tree = function
  | [] -> Empty
  | hd :: tl -> extend hd (list2tree tl)
