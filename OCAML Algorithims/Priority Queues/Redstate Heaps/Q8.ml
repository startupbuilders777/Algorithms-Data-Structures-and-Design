module type OrderedType = sig
  type t
  val compare : t -> t -> int
end

module OrderedInt : (OrderedType with type t = int) =
  struct 
    type t = int
    let compare x y = if(x < y) then -1 else if (x = y) then 0 else 1
  end

module type mPQ = sig
  module Elem : OrderedType
 
  type 'a bintree =
    Empty
  | Node of int * 'a * 'a bintree * 'a bintree
 
  type mpq = Elem.t bintree
 
  (* type mpq *)
 
  val empty : mpq
  val isEmpty : mpq -> bool
  val merge : mpq -> mpq -> mpq
  val insert : Elem.t -> mpq -> mpq
  val findMin : mpq -> Elem.t option
  val deleteMin : mpq -> mpq option
end
 

module RsHeapmPQ (E : OrderedType) : (mPQ with module Elem = E) =
struct
  module Elem = E

   type 'a bintree =
    Empty
  | Node of int * 'a * 'a bintree * 'a bintree
 
  type mpq = Elem.t bintree
 
  let empty : mpq = Empty;; 
  
  let isEmpty (mhpq: mpq) : bool =
      match mhpq with
      | Empty -> true
      | _ -> false;;


(*

QUESTION:

In this question, you will refine the idea from maxiphobic heaps 
of using tree size to bound recursion depth 
for merging (and thus for other operations).

A redstate heap is either empty, or the right subtree has at least 
as many nodes as the left subtree, and both subtrees are redstate heaps.

Work out an algorithm for merging two redstate heaps. In a comment in 
your program, provide an analysis that demonstrates that your merge 
operation takes O(logn)O(log⁡n) time when the result has nn nodes. 
What is the base of the logarithm (in the depth of the recursion) 
and how does it compare to maxiphobic heaps?

*)

(*

ANSWER:

I place the invarient that each redstate heap is near perfect
(all leaves have the same depth or depth-1).
So the leftSubtree has depth d or d+1 while the right subtree has depth d.
When a two trees are being merged, decompose each tree into its roots, 
and left and right subtrees. THe new root of the merged tree is the smallest of the two roots.
The root that wasnt picked, denote X. 
Take the left subtree of one tree (denote the depth a or a+1) and 
recursively merge it with the right subtree of the other tree(denote the depth b).
Take the right subtree of one tree(denote the depth a) and recursively merge it with 
the left subtree of the other tree(denote the depth b or b+1).

The newly merged left subtree will have depth(a + b or (a + 1) + b).
The new merged right subtree will have depth (a + b or a + (b + 1)).

Insert the root x on the side with the smaller depth.

*)
    let rec merge (leftMPQ: mpq) (rightMPQ: mpq): mpq = 
        let rec mergeBranches (topLevelRoot : Elem.t) (otherRoot : Elem.t) 
                              (ll: mpq) (lr: mpq) (rl: mpq) (rr: mpq) (totalSize : int) : mpq = 
            let getSize tree =
                match tree with
                | Node(s, _, _, _) -> s
                | Empty -> 0 in
            let newLeft = merge(ll)(rr) in
            let newRight = merge(lr)(rl) in
            let newLeftSize = getSize(ll) + getSize(rr) in 
            let newRightSize = getSize(lr) + getSize(rl) in
            let otherRootTree = Node(1, otherRoot, Empty, Empty) in 
            if(newLeftSize >= newRightSize) then Node(totalSize, topLevelRoot, newLeft, merge(otherRootTree)(newRight))
            else Node(totalSize, topLevelRoot, merge(otherRootTree)(newLeft), newRight) in    
        match (leftMPQ, rightMPQ) with
        | (Empty, Empty) -> Empty
        | (_, Empty) -> leftMPQ
        | (Empty, _) -> rightMPQ
        | (Node(sizeLeft, lVal, ll, lr), Node(sizeRight, rVal, rl, rr)) -> 
            let totalSize = sizeLeft + sizeRight in
            if (lVal <= rVal) then mergeBranches(lVal)(rVal)(ll)(lr)(rl)(rr)(totalSize)
            else mergeBranches(rVal)(lVal)(ll)(lr)(rl)(rr)(totalSize) ;;                                                             

let rec insert (element: Elem.t) (mhpq: mpq) : mpq =
    let elementTree = Node(1, element, Empty, Empty) in
    match mhpq with
    | Empty -> elementTree
    | Node (_, _, _, _) -> merge(elementTree)(mhpq);;

let findMin (mhpq: mpq) : Elem.t option =
    match mhpq with 
    | Node(_, v, _, _) -> Some v
    | Empty -> None;;


let deleteMin (mhpq: mpq) : mpq option =
  match mhpq with
  | Empty -> None
  | Node(_, _, left, right) -> Some(merge(left)(right)) ;;

end

