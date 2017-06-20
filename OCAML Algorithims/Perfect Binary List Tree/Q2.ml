type 'a tree = Leaf of 'a | Node of int * 'a tree * 'a tree
type 'a digit = Zero | One of 'a tree

type 'a sequence = 'a digit list

let empty = [];;

let isEmpty lst =
  match lst with  
  | [] -> true
  | _ -> false;;

let rec first seq =
  let rec getFirstElementInTree tree =
    match tree with
    | Leaf(a) -> a
    | Node(_, l, r) -> getFirstElementInTree l in
  match seq with 
  | [] -> None
  | Zero::tl -> first tl
  | One(tree) :: _ -> Some (getFirstElementInTree tree);;

let rest seq = 
  let rec containsOne seq = 
      match seq with
      | [] -> false
      | One(a)::tl -> true
      | _ :: tl -> containsOne tl in 
  let rec splitTreeAndRemoveFirst tree splitCounter acc =
    match (tree, splitCounter) with
    | (_, 0) -> acc
    | (Node(_, lt, rt), _) -> splitTreeAndRemoveFirst(lt)(splitCounter - 1)(One(rt)::acc) in
  let rec aux seq splitCounter=
    match seq with 
    | [] -> None
    | Zero :: tl -> aux(tl)(splitCounter + 1)
    | One(tree) :: tl -> 
                        let split = splitTreeAndRemoveFirst(tree)(splitCounter) in
                         Some(
                            if (containsOne tl) then  split([Zero]) @ tl
                            else split([]) @ tl
                             )
                        in
  aux seq 0;;

let extend ele seq = 
  let rec combineTrees element rightTree =
    match (element, rightTree) with
    | (Leaf(a), Leaf(b))  -> Node(2, Leaf(a), Leaf(b))
    | (Node(_, _, _), Node(sizeR, _, _)) -> Node(sizeR*2, element, rightTree) in
  let rec aux element seq =
    match seq with 
    | [] -> One(element)::[]
    | Zero :: tl -> One(element)::tl
    | One(tree) :: tl -> Zero::aux((combineTrees(element)(tree)))(tl)  in  
  aux (Leaf (ele)) seq;;

let emptyPBLT = []

let simplePBLT : string sequence = [ One(Leaf("b"))];;

let pblt : string sequence = [  
  Zero ; 
  One(Node(2, Leaf("a"), Leaf("b"))) ; 
  Zero ;   
  Zero ; 
  One(Node(16, 
           Node(8, 
                Node(4, 
                     Node(2, Leaf("c"), Leaf("d")), 
                     Node(2, Leaf("e"), Leaf("f"))), 
                Node(4, 
                     Node(2, Leaf("g"), Leaf("h")),
                     Node(2, Leaf("i"), Leaf("j")))),
           Node(8, 
                Node(4, 
                     Node(2, Leaf("k"), Leaf("l")), 
                     Node(2, Leaf("m"), Leaf("n"))), 
                Node(4, 
                     Node(2, Leaf("o"), Leaf("p")), 
                     Node(2, Leaf("q"), Leaf("r"))))) 
     )];;


exception INDEX_OUT_OF_BOUNDS;;

let index (index : int) (sequence : 'a sequence) : 'a option =
  let rec indexPerfectBinaryTree index l r =
    match (l, r) with 
    | (Leaf(lvalue), Leaf(rvalue)) when index = 0 -> lvalue
    | (Leaf(lvalue), Leaf(rvalue)) when index = 1 -> rvalue
    | (Node(sizeL, ll, lr), Node(_,_,_)) when index < sizeL -> (indexPerfectBinaryTree index ll lr)
    | (Node(_,_,_), Node(sizeR, rl,rr))  -> (indexPerfectBinaryTree (index - sizeR) rl rr)
  in
  let rec indexPerfectBinaryList index sequence = 
    match sequence with 
    | Zero :: tl -> indexPerfectBinaryList index tl 
    | One(tree) :: tl -> 
      (match tree with
       | Leaf (value)  -> if index = 0 then value 
         else indexPerfectBinaryList (index - 1) tl
       | Node (size, l, r) -> if index < size then indexPerfectBinaryTree index l r
         else indexPerfectBinaryList (index - size) tl
      )
    | _ -> raise INDEX_OUT_OF_BOUNDS
  in
  if index < 0 || sequence = []
  then None 
  else
    try Some (indexPerfectBinaryList index sequence) with
    | INDEX_OUT_OF_BOUNDS -> None


(*
Running Times:
////////////////////////////////////////////////////////////////////////////
Operation isEmpty : Performance = O(1)
isEmpty has a running time of O(1) since it is a single operation 
that checks if the list is empty or not.
////////////////////////////////////////////////////////////////////////////
Operation extend : : Performance =  O(log(n))
Let n be the number of elements in the sequence.
The list has log(n) elements because each element in the list is either a zero 
or a binary tree that contains 2^n elements of the sequence. 

Given an element and a seq with n elements, the list is iterated to add the element. 
The worst case of iterating through the tree is O(log (n)) because the list is length log n.
When an element needs to be added where a tree is located,  
the perfect binary trees need to be shifted to the right 
and merged to make space for the element. The shifting of trees to the right is
done by calling the combineTrees which runs in constant time.    
CombineTrees is called when there is a sequence of Ones that need to be combined in the list,
and then is terminated when that sequence of Ones containing binary trees has been combined.
There will be at most log(n) ones in the list to combine, so that operation will take 
O(1) * log(n) = O(log(n)).

Therefore the total worst case operations is the number of worst case operations to iterate 
and the number of worst case operations to combine the trees which is 
O(log(n)) + O(log(n)) = O(log(n))
////////////////////////////////////////////////////////////////////////////
Operation first : Performance O(log(n)
Let n be the number of elements in the sequence.
The list has log(n) elements because each element in the list is either a zero 
or a binary tree that contains 2^n elements of the sequence. 

first recursively looks for the element by going through the list to find 
the first tree that contains elements which in the worst case will take O(log(n)), 
and then goes down that tree to the find the leftmost leaf of that tree (the height of the tree will be log(n)).
So the performance of first is O(log(n)) + O(log(n)) = O(log(n))  
////////////////////////////////////////////////////////////////////////////
Operation rest: Performance O(log(n))
Let n be the number of elements in the sequence.
The list has log(n) elements because each element in the list is either a zero 
or a binary tree that contains 2^n elements of the sequence.

The rest operation iterates through the list and removes the element from the first tree found.
The list iteration is worst case O(log(n)) since the list has length log n.
When a tree is found, it is recursively broken down to smaller sub trees. Worst case, the tree
has to be broken down log(n) times because the height would be log(n).
An accumulator is used to accumate these split trees and appended them to the rest of the list.
The append operation is log(n), because the number of split trees to append to the list 
can be at most log(n). A check needs to be made to see if the sequence contains a One 
right before split trees is called, and that operation takes O(log(n)) because the sequence
worst case has length log(n).

The entire operation worst case is O(log(n)), because it is composed of a sum of processes which 
are slowest O(log(n)).
////////////////////////////////////////////////////////////////////////////
Operation index : Performance O(log(n))
Let n be the number of elements in the sequence.
The list has log(n) elements because each element in the list is either a zero 
or a binary tree that contains 2^n elements of the sequence.

The list is iterated until the tree that contains the element is found. 
This will take O(log(n)) worst case (this occurs when the tree is the last tree in the list). 
Then the tree is searched recursively, going down a right or left child node on 
each recursive call until the element being indexed is found. 

This will take O(log(n)) since the height of the tree is at most log(n).

Therefore the operation is O(log(n)) + O(log(n)) = O(log(n))
////////////////////////////////////////////////////////////////////////////
*)

let() = assert(isEmpty emptyPBLT = true)
let() = assert(isEmpty simplePBLT = false)
let() = assert(isEmpty pblt = false)
let() = assert(index (-1) pblt = None)
let() = assert(index 0 pblt = Some "a")
let() = assert(index 1 pblt = Some "b")
let() = assert(index 2 pblt = Some "c")
let() = assert(index 3 pblt = Some "d")
let() = assert(index 4 pblt = Some "e")
let() = assert(index 5 pblt = Some "f")
let() = assert(index 6 pblt = Some "g")
let() = assert(index 7 pblt = Some "h")
let() = assert(index 8 pblt = Some "i")
let() = assert(index 9 pblt = Some "j")
let() = assert(index 10 pblt = Some "k")
let() = assert(index 11 pblt = Some "l")
let() = assert(index 12 pblt = Some "m")
let() = assert(index 13 pblt = Some "n")
let() = assert(index 14 pblt = Some "o")
let() = assert(index 15 pblt = Some "p")
let() = assert(index 16 pblt = Some "q")
let() = assert(index 17 pblt = Some "r")
let() = assert(index 18 pblt = None)
let() = assert( (extend "a" 
                   (extend "b" 
                      (extend "c" 
                         (extend "d" 
                            (extend "e" 
                               (extend "f" 
                                  (extend "g" 
                                     (extend "h" 
                                        (extend "i" 
                                           (extend "j" 
                                              (extend "k" 
                                                 (extend "l" 
                                                    (extend "m" 
                                                       (extend "n" 
                                                          (extend "o" 
                                                             (extend "p" 
                                                                (extend "q" 
                                                                   (extend "r" [])

                                                                ))))))))))))))))) = pblt)
let get = function
  | Some x -> x;;

let() = assert(  (extend "c" 
                    (extend "d" 
                       (extend "e" 
                          (extend "f" 
                             (extend "g" 
                                (extend "h" 
                                   (extend "i" 
                                      (extend "j" 
                                         (extend "k" 
                                            (extend "l" 
                                               (extend "m" 
                                                  (extend "n" 
                                                     (extend "o" 
                                                        (extend "p" 
                                                           (extend "q" 
                                                              (extend "r" [])))))))))))))))) = (get(rest(get(rest(pblt))))) )

let() = assert(  (extend "b" (extend "c" 
                                (extend "d" 
                                   (extend "e" 
                                      (extend "f" 
                                         (extend "g" 
                                            (extend "h" 
                                               (extend "i" 
                                                  (extend "j" 
                                                     (extend "k" 
                                                        (extend "l" 
                                                           (extend "m" 
                                                              (extend "n" 
                                                                 (extend "o" 
                                                                    (extend "p" 
                                                                       (extend "q" 
                                                                          (extend "r" []))))))))))))))))) = (get(rest(pblt))) )


let() = assert(   (extend "m" 
                     (extend "n" 
                        (extend "o" 
                           (extend "p" 
                              (extend "q" 
                                 (extend "r" [])))))) = (get(rest
                                                               (get(rest
                                                                      (get(rest
                                                                             (get(rest
                                                                                    (get(rest
                                                                                           (get(rest
                                                                                                  (get(rest
                                                                                                         (get(rest
                                                                                                                (get(rest
                                                                                                                       (get(rest
                                                                                                                              (get(rest
                                                                                                                                     (get(rest(pblt))))))))))))))))))))))))) )