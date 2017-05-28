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

Operation isEmpty : 'a sequence -> bool
isEmpty has a running time of O(1) since it is a single operation that checks if the list is empty or not.


Operation extend : 'a -> 'a sequence -> 'a sequence
Given an element and a seq of size n, 
extend iterates through the seq, combining trees recursively in the process to form bigger 
trees that can hold the extended element.
Combine trees takes an element 

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



Operation first : 'a sequence -> 'a option
let rec first seq =
  let rec getFirstElementInTree tree =
    match tree with
    | Leaf(a) -> a
    | Node(_, l, r) -> getFirstElementInTree l in
  match seq with 
  | [] -> None
  | Zero::tl -> first tl
  | One(tree) :: _ -> Some (getFirstElementInTree tree);;


Opeartion rest : 'a sequence -> 'a sequence option 
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


Operation index : int -> 'a sequence -> 'a option
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