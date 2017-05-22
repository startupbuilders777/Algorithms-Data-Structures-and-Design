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
    let rec splitTreeAndRemoveFirst tree splitCounter acc =
        match (tree, splitCounter) with
        | (_, 0) -> acc
        | (Node(_, lt, rt), _) -> splitTreeAndRemoveFirst(lt)(splitCounter - 1)(One(rt)::acc) in
    let rec aux seq splitCounter=
        match seq with 
        | [] -> None
        | Zero :: tl -> aux(tl)(splitCounter + 1)
        | One(tree) :: tl -> Some(splitTreeAndRemoveFirst(tree)(splitCounter)([Zero]) @ tl)in  
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