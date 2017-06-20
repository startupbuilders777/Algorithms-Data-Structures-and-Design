

(*
Using unified tree model
*)


type 'a treap =
  | Empty
  | Node of int * 'a * 'a treap * 'a treap ;;

(*
The int is a priority, 
the a has the value of the key.


type 'a bst = 'a treap

val empty : 'a bst
val unwrap : 'a bst -> ('a * 'a bst * 'a bst) option
val join : 'a bst -> 'a -> 'a bst -> 'a bst



module type bst_impl = sig
  type 'a bst
  val empty : 'a bst
  val unwrap : 'a bst -> ('a * 'a bst * 'a bst) option
  val join : 'a bst -> 'a -> 'a bst -> 'a bst
end


Provide an OCaml implementation of derived functions within the unified framework discussed in lecture, conforming to this signature:

module type bst_deriv = sig

  module BSTImpl : bst_impl

  type 'a bst = 'a BSTImpl.bst

  val split : 'a -> 'a bst -> 'a bst * bool * 'a bst
  val is_empty : 'a bst -> bool
  val insert : 'a -> 'a bst -> 'a bst
  val find_max : 'a bst -> 'a option
  val delete_max : 'a bst -> ('a * 'a bst) option
  val join2 : 'a bst -> 'a bst -> 'a bst
  val delete : 'a -> 'a bst -> 'a bst
  val union : 'a bst -> 'a bst -> 'a bst
  val intersect : 'a bst -> 'a bst -> 'a bst
  val diff : 'a bst -> 'a bst -> 'a bst
end

module BSTDeriv (I : bst_impl) : (bst_deriv with module BSTImpl = I)

find_max uses structural recursion; insert, delete_max, join2, and delete 
should not use recursion (join2 is like join, but without the in-between element). 
All operations should take time logarithmic in the number of nodes of the tree consumed 
or produced, assuming that the implementation of join conforms to the 
unified framework analysis discussed in lecture.

Consider using not only your Q9 implementation to test your Q10 code, 
but another balanced BST implementation as well.
*)

let empty treap = 
  match treap with
  | Empty -> true
  | _ -> false;;

(*When do you assign the randomized priority to the element*)
let rec join treapL element treapR =
  let getKey element = 
    match element with 
    | (pri, key) -> key in
  let getPriority element = 
    match element with 
    | (pri, key) -> pri in
  let elementPriority = getPriority(element) in
  let elementKey = getKey(element) in 
  match (treapL, treapR) with
  | (Empty, Empty) -> Node(elementPriority, elementKey, Empty, Empty)
  | (Node(priL, keyL, ll, lr), Node(priR, keyR, rl, rr)) -> 
    if(elementPriority <= priL && elementPriority <= priR) then Node(elementPriority, elementKey, treapL, treapR)
    else if(priL <= elementPriority && priL <= priR) then Node(priL, keyL, ll, join(lr)(element)(treapR))
    else Node(priR, keyR, join(treapL)(element)(rl), rr)
  | (Node(priL, keyL, ll, lr), Empty) -> 
    if(elementPriority <= priL) then Node(elementPriority, elementKey, treapL, treapR)
    else Node(priL, keyL, ll, join(lr)(element)(Empty))
  | (Empty, Node(priR, keyR, rl, rr)) -> 
    if(elementPriority <= priR) then Node(elementPriority, elementKey, treapL, treapR)
    else Node(priR, keyR, join(Empty)(element)(rl), rr) ;;


let rec split element treap =
  let getKey element = 
    match element with 
    | (priority, key) -> key in
  let elementKey = getKey(element) in  
  match treap with 
  | Empty -> (Empty, false, Empty)
  | Node(priority, key, left, right) -> 
    let treapRootElement = (priority, key) in
    if (elementKey = key) then (left, true, right)
    else if (elementKey > key) then 
      let (lprime, b, rprime) = split(element)(right) in
      (join(left)(treapRootElement)(lprime), b, rprime)
    else 
      let (lprime, b, rprime) = split(element)(left) in 
      (lprime, b, join(rprime)(treapRootElement)(right)) ;;


let isEmpty treap =
  match treap with 
  | Empty -> true
  | _ -> false;;

let insert element treap = 
  let (left, b, right) = split(element)(treap) in
  join(left)(element)(right);;

let find_max treap =
  let rec aux treap = 
    match treap with 
    | Node(pri, key, Empty, Empty) -> (pri, key)
    | Node(_, _, left, right) -> aux(right) in
  match treap with
  | Empty -> None
  | _ -> Some(aux(treap)) ;;

let delete_max_normal treap =
  let rec aux treap = 
    match treap with 
    | Node(pri, key, Empty, Empty) -> ((pri, key), Empty)
    | Node(pri, key, left, right) -> let (e, tree) = aux(right) in (e, Node(pri, key, left, tree)) in 
  match treap with 
  | Empty -> None
  | _ -> Some(aux(treap)) 

let delete_max_with_split treap = 
  match find_max(treap) with 
  | None -> None
  | Some x -> let (left, b, right) =  split(x)(treap) in Some(x, left);;

let rec union treapL treapR =
  let getRootOfTreap treap = 
    match treap with
    | Node(pri, key, left, right) -> (pri, key) in
  match (treapL, treapR) with
  | (Empty, Empty) -> Empty
  | (treapL, Empty) -> treapL
  | (Empty, treapR) -> treapR
  | (treapL, Node(priR, keyR, rl, rr)) -> 
    let rRoot = getRootOfTreap(treapR) in
    let (lprime, b, rprime) = split(rRoot)(treapL) in join(union(lprime)(rl))(rRoot)(union(rprime)(rr));;

let rec getHeight treap =
  let max a b =
    if(a >= b) then a
    else b in
  match treap with 
  | Empty -> 0
  | Node(_,_, left, right) -> max(1 + getHeight(left))(1 + getHeight(right)) ;;

let rec countElements bintree = 
  match bintree with
  | Empty -> 0
  | Node(_, _, left, right) -> 1 + countElements(left) + countElements(right);;
(*
let rec join2 treap1 treap2 = (*keep the other element, keep all the elements*)
  let getRootOfTreap treap = 
    match treap with
    | Node(pri, key, left, right) -> (pri, key) in
  match(treap1, treap2) with 
  | (Empty, Empty) -> Empty
  | (Empty, treap2) -> treap2
  | (treap1, Empty) -> treap1
  | (treap1, Node(rPri, rKey, left, right)) -> 
  let rRoot = getRootOfTreap(treap2) in
  let (lprime, b, rprime) = split(rRoot)(treap1) in
  if(b = true) then 
  else join(union(lprime)(left))(rRoot)(union(rprime)(right))
*)

(*Minium in terms of priority*)
let rec getMinimum treap = 
  match treap with 
  | Empty -> None
  | Node(pri, value, _, _) -> Some (pri, value);; 
(*
let rec join2 treap1 treap2 = (*keep the other element, keep all the elements*)
  let getRootOfTreap treap = 
    match treap with
    | Node(pri, key, left, right) -> (pri, key) in
  match(treap1, treap2) with 
  | (Empty, Empty) -> Empty
  | (Empty, treap2) -> treap2
  | (treap1, Empty) -> treap1
  | (treap1, Node(rPri, rKey, left, right)) -> 
  let rRoot = getRootOfTreap(treap2) in
  let (lprime, b, rprime) = split(rRoot)(treap1) in
  if(b = true) then insert(rRoot)(join(union(lprime)(left))(rRoot)(union(rprime)(right)))
  else join(union(lprime)(left))(rRoot)(union(rprime)(right));;
*)

let delete element treap =
  let (left, b, right) = split(element)(treap) in union(left)(right);;

(*Assuming all elements are distict*)
let rec treapToList treap = 
  match treap with 
  | Empty -> []
  | _ -> let Some(minimum) = getMinimum(treap) in minimum::treapToList(delete(minimum)(treap));;

let rec treapToList2 treap = 
  match treap with 
  | Empty -> []
  | _ -> let Some(max, tree) = delete_max_with_split(treap) in max::treapToList(tree);;

let rec intersection treap1 treap2 = 
  let getRootOfTreap treap = 
    match treap with
    | Node(pri, key, left, right) -> (pri, key) in
  match (treap1, treap2) with 
  | (Empty, Empty) -> Empty
  | (Empty, treap2) -> Empty
  | (treap1, Empty) -> Empty
  | (treap1, Node(_,_, left, right)) ->
    let rRoot = getRootOfTreap(treap2) in
    let (lprime, b, rprime) = split(rRoot)(treap1) in
    if(b = true) then join(intersection(lprime)(left))(rRoot)(intersection(rprime)(right))
    else union(intersection(lprime)(left))(intersection(rprime)(right));;

let rec diff treapA treapB =
  let getRootOfTreap treap = 
    match treap with
    | Node(pri, key, left, right) -> (pri, key) in
  match (treapA, treapB) with
| (Empty, Empty) -> Empty
| (treapA, Empty) -> treapA
| (Empty, treapB) -> Empty
| (treapA, Node(_,_, left, right)) ->
  let bRoot = getRootOfTreap(treapB) in
  let (lprime, b, rprime) = split(bRoot)(treapA) in union(diff(lprime)(left))(diff(rprime)(right))  ;;



Random.self_init();;
let e1 = (Random.int 90000, 1);;
let e2 = (Random.int 90000, -1);;
let e3 = (Random.int 90000, 6);;
let e4 = (Random.int 90000, -7);;
let e5 = (Random.int 90000, 3);;
let e6 = (Random.int 90000, 5);;
let e7 = (Random.int 90000, 2);;
let e8 = (Random.int 90000, 9);;

let e9 = (Random.int 90000, 15);;
let e10 = (Random.int 90000, -4);;
let e11 = (Random.int 90000, 12);;
let e12 = (Random.int 90000, -1);;
let e13 = (Random.int 90000, 8);;
let e14 = (Random.int 90000, 4);;
let e15 = (Random.int 90000, 3);;
let e16 = (Random.int 90000, 2);;


let treap1 = insert(e8)(insert(e7)(insert(e6)(insert(e5)(insert(e4)(insert(e3)(insert(e2)(insert(e1)(Empty))))))));;
let treap2 = insert(e9)(insert(e10)(insert(e11)(insert(e12)(insert(e13)(insert(e14)(insert(e15)(insert(e16)(Empty))))))));;

let treap3 = union(treap1)(treap2);;
(*
utop # treap1;;

Node (188, -1, 
  Node (2931, -7, Empty, Empty), 
  Node (27974, 9, 
    Node (35965, 1, Empty, 
                    Node (54055, 3, Node (55857, 2, Empty, Empty), 
                                    Node (60938, 5, Empty, 
                                                    Node (70431, 6, Empty, Empty)))), 
          Empty))      

utop # treap2;;
Node (14951, -4, Empty, 
                 Node (28325, 15, 
                                  Node (47503, 3, 
                                                  Node (71454, 2, 
                                                                  Node (73634, -1, Empty, Empty), 
                                                                  Empty), 
                                                  Node (52183, 4, Empty, 
                                                                  Node (82753, 8, Empty, 
                                                                                  Node (89682, 12, Empty, Empty)))), Empty))   

utop # treap3;;
Node (2931, -7, Empty, 
                Node (14951, -4, Empty,                                                                                                                                                                                    
                                 Node (27974, 9, Node (35965, 1, Empty, Node (52183, 4, Empty, Node (60938, 5, Empty, Node (70431, 6, Empty, Node (82753, 8, Empty, Empty))))), Node (28325, 15, Node (89682, 12, Empty, Empty), Empty))))

*)
(*let treap1 = insert(3, ) *)

(*

(*
  let join2 : 'a bst -> 'a bst -> 'a bst
  *)
  *)