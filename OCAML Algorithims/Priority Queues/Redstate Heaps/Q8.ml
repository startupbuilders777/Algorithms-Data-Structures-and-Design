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
operation takes O(logn) time when the result has nn nodes. 
What is the base of the logarithm (in the depth of the recursion) 
and how does it compare to maxiphobic heaps?

*)

(*
ANSWER:
This merge has an efficiency of O(log n) time. The log in this efficiency has base 2. 
There are 2 cases handled by this merge. 
In both cases, the 2 trees (call them tree X and tree Y) will 
be split into their subtrees and their roots. (call the roots x and y) 
The smaller root, let's assume that is x, will be the new root for the merged tree. 
We are left now with the left and right subtree of X, and the tree Y. 
The left and right subtrees of X and tree Y, which is a total of 3 trees, 
have to be merged in such a way that efficiency remains O(log n).
Lets call these three trees that we are merging to be A, B, C, which along with the 
smallest root will form the merged tree. 

Pick the largest tree from A, B, C. Lets assume that is tree C. 
Also let S = size(A) + size(B) + size(C). 
The size of C >= A. 
The size of C >= B. 
Therefore 2C >= A + B.
S = A + B + C implies S - C = A + B 
=> 2C >= S - C 
=> 3C >= S
=> C >= (1/3)S

In other words, the size of the biggest tree, C will always be greater than (1/3)rd the Size of S.
When C is (1/3)rd the size of S, A+B is (2/3)rd the size of S. 
Using the maxiphobic algorithm of combining the two smaller subtrees and putting
that as the left subtree for the merged tree will be bad for the recursion when A+B is (2/3)rd the size of S
because this will mean we will have log n with base 3/2 recurive calls. 
However, the maxiphobic algorithm is a good algorithm, when the size of C is greater than or equal to (1/2)S 
because that means the size of (A+B) is less than or equal to (1/2)S, so when A + B is recursively merged, 
there is log n with base 2 recursive calls (if this case holds for each recursive call). 

Therefore, during runtime, we will check if the size of the biggest tree is greater than or equal to (1/2) S
and if it is, we will use the maxiphobic algorithm. 
Otherwise, we will have to devise a merge for the case when there is between 1/3 to 1/2 nodes in the biggest 
tree because this means that (A+B) > (1/2) S nodes and that will reduce the efficiency of recursive merge. 
We have to ensure that the recursive merge will be applied on 2 trees that have  less than or equal to (1/2) S nodes
to have log n base 2 efficiency. 

In this second case where C has less than (1/2) S nodes, we can split C into CL, CR and rootC. 
CR will have atleast as many nodes as CL by the redstone heap invarient, so CR >= CL. 
Therefore, CL will be one of the candidates for the recursive merge operation.
The biggest CL can be is (1/4) of S, because the biggest C can be in this case is (1/2) S nodes.
The math is:

CR >= CL 
CR + CL = C => CR = C - CL
C - CL >= CL 
C >= 2CL 
CL <= C/2. 
Also in this case, C <= (1/2)S So,
CL <= (1/4)S

Also  A+B >= (1/2) S . Assume A < B.
So A is the smaller tree. Since A is smaller than B, A can have at most size (1/4) S.
SO merging CL with A will result in a size of less than (1/2) S (Since CL <= (1/4)S and A <= (1/4)S )
and this will constrain the other cases merge operation to log n base 2.
The other two trees, CR and B, will use the Node constructor to create 
the right tree for the new merged tree which will take O(1).

Therefore in both cases, the efficieny is O(log n) base 2, so this merge operation 
has O(log n) base 2.
*) 

  let rec merge (leftMPQ : 'c bintree) (rightMPQ : 'c bintree) : 'c bintree = 
    let getSize tree =
      match tree with
      | Node(s, _, _, _) -> s
      | Empty -> 0 in
    let rec mergeBranches topLevelRoot a b cBigTree totalSize = 
      match (getSize(cBigTree) >= totalSize/2) with 
      | true -> Node(totalSize, topLevelRoot, merge(a)(b) , cBigTree)
      | false -> (
          match cBigTree with 
          | Node(sizecBigTree, cRoot, cl, cr) -> (
            match getSize(a) <= getSize(b) with
            | true -> 
            (match b with 
              | Node(bSize, bRoot, _, _) -> (
                if(bRoot <= cRoot && bSize <= getSize(cr)) then Node(totalSize, topLevelRoot, merge(a)(cl), Node(bSize + getSize(cr)+1, bRoot, b, cr))
                else if(bRoot <= cRoot && bSize > getSize(cr)) then Node(totalSize, topLevelRoot, merge(a)(cl), Node(bSize + getSize(cr)+1, bRoot, cr, b))
                else if(bRoot > cRoot && bSize > getSize(cr)) then Node(totalSize, topLevelRoot, merge(a)(cl), Node(bSize + getSize(cr)+1, cRoot, cr, b))
                else Node(totalSize, topLevelRoot, merge(a)(cl), Node(bSize + getSize(cr)+1, cRoot, cr, b))
              )
              | Empty -> Node(totalSize, topLevelRoot, merge(a)(cl), Node(totalSize,cRoot, cr, Empty) )
            )
            | false -> 
            ( match a with 
              | Node(aSize, aRoot, _, _) -> (
                if(aRoot <= cRoot && aSize <= getSize(cr)) then Node(totalSize, topLevelRoot, merge(b)(cl), Node(aSize + getSize(cr)+1, aRoot, a, cr))
                else if(aRoot <= cRoot && aSize > getSize(cr)) then Node(totalSize, topLevelRoot, merge(b)(cl), Node(aSize + getSize(cr)+1, aRoot, cr, a))
                else if(aRoot > cRoot && aSize > getSize(cr)) then Node(totalSize, topLevelRoot, merge(b)(cl), Node(aSize + getSize(cr)+1, cRoot, cr, a))
                else Node(totalSize, topLevelRoot, merge(b)(cl), Node(aSize + getSize(cr), cRoot, cr, a))
              )
              | Empty -> Node(totalSize, topLevelRoot, merge(b)(cl), Node(getSize(cr) + 1, cRoot, cr, Empty))   
            )
            )
          | Empty -> Node(totalSize, topLevelRoot, Empty, Empty) 
      ) in
match (leftMPQ, rightMPQ) with
| (Empty, Empty) -> Empty
| (_, Empty) -> leftMPQ
| (Empty, _) -> rightMPQ
| (Node(sizeLeft, lVal, ll, lr), Node(sizeRight, rVal, rl, rr)) -> 
  let totalSize = sizeLeft + sizeRight in
  if (lVal <= rVal) then 
    begin
      let sizeLL = getSize(ll) in
      let sizeLR = getSize(lr) in
      let sizeR = getSize(rightMPQ) in
      if(sizeLL >= sizeLR && sizeLL >= sizeR) then mergeBranches(lVal)(rightMPQ)(lr)(ll)(totalSize)
      else if(sizeLR >= sizeLL && sizeLR >= sizeR) then mergeBranches(lVal)(ll)(rightMPQ)(lr)(totalSize)
      else mergeBranches(lVal)(ll)(lr)(rightMPQ)(totalSize)
    end
  else 
    begin
      let sizeRL = getSize(rl) in
      let sizeRR = getSize(rr) in
      let sizeL = getSize(leftMPQ) in
      if(sizeRL >= sizeRR && sizeRL >= sizeL) then mergeBranches(rVal)(leftMPQ)(rr)(rl)(totalSize)
      else if(sizeRR >= sizeRL && sizeRR >= sizeL) then mergeBranches(rVal)(rl)(leftMPQ)(rr)(totalSize)
      else mergeBranches(rVal)(rl)(rr)(leftMPQ)(totalSize)                                                  
    end

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

