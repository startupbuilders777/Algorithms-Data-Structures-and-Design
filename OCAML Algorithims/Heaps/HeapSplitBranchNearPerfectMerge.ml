type 'a bintree =
    Empty
  | Node of int * 'a * 'a bintree * 'a bintree

let rec merge leftMPQ rightMPQ = 
        let rec mergeBranches  topLevelRoot otherRoot ll  lr rl rr totalSize = 
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

let rec insert element mhpq =
    let elementTree = Node(1, element, Empty, Empty) in
    match mhpq with
    | Empty -> elementTree
    | Node (_, _, _, _) -> merge(elementTree)(mhpq);;

let findMin mhpq =
    match mhpq with 
    | Node(_, v, _, _) -> Some v
    | Empty -> None;;


let deleteMin mhpq  =
  match mhpq with
  | Empty -> None
  | Node(_, _, left, right) -> Some(merge(left)(right)) ;;

let tree1 = insert(-1)(insert(-2)(insert(6)(insert(4)(insert(2)(insert(1)(Empty))))));;
let tree2 = insert(3)(insert(-99)(insert(2)(insert(1)(Empty))));;
let tree3 insert(-4)(tree2);;
(*
tree1:
Node (6, -2, 
            Node (2, 1, 
                        Empty, 
                        Node (1, 4, 
                                    Empty, 
                                    Empty)), 
            Node (3, -1, 
                        Node (1, 6, 
                                    Empty, 
                                    Empty), 
                        Node (1, 2, 
                                    Empty, 
                                    Empty)))

tree2:  
Node (4, -99, Node (1, 1, 
                            Empty, 
                            Empty), 
              Node (2, 2,   Empty, 
                            Node (1, 3, 
                                        Empty, 
                                        Empty))) 

val tree3  
Node (5, -99, 
             Node (2, 2, 
                        Empty, 
                        Node (1, 3, 
                                    Empty, 
                                    Empty)), 
             Node (2, -4, 
                        Empty, 
                        Node (1, 1, 
                                    Empty, 
                                    Empty))) 

merge tree2 with tree3:

Node (9, -99, 
                Node (4, -99, 
                             Node (1, 1, 
                                        Empty, 
                                        Empty),
                             Node (2, -4, 
                                        Empty, 
                                        Node (1, 1, Empty, Empty))), 
                Node (4, 2, 
                            Node (1, 3, Empty, 
                                        Empty), 
                            Node (2, 2, Empty, 
                                        Node (1, 3, Empty, Empty))))
*)