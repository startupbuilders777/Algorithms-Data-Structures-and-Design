    let compare x y = if(x < y) then -1 else if (x = y) then 0 else 1

type 'a brt =
        Empty
        | Node of 'a * 'a brt * 'a brt
    
  type pq = int brt;; 
  
  let empty : pq = Empty;; 
  
  let isEmpty (bhpq: pq) : bool =
      match bhpq with
      | Empty -> true
      | _ -> false;;

  let rec insert (element: int) (bhpq: pq) : pq =
      match bhpq with
      | Empty -> Node(element, Empty, Empty) 
      | Node (v, left, right) -> if (compare(element)(v)) = -1 then Node(element, insert(v)(right), left) 
                              else Node(v, insert(element)(right), left) ;;

  let findMin (bhpq: pq)  =
      match bhpq with 
      | Node(v, left, right) -> Some v
      | Empty -> None;;

  let getFirstElementFromPair pair =
    match pair with
    | (first, second) -> first;;
  let getSecondElementFromPair pair = 
    match pair with
    | (first, second) -> second;;

  let rec deleteOne (bhpq) = 
    match bhpq with
    | Node(v, Empty, Empty) -> (v, Empty)
    | Node(v, left, right) ->  let result = deleteOne(left) in
                               let element = getFirstElementFromPair(result) in
                               let tree = getSecondElementFromPair(result) in
                               (element, (Node(v, right, tree))) ;;

let braunHeapPQ = insert(0)(insert(-3)(insert(99)(insert(5)(insert(12)(insert(7)(insert(0)(insert(3)(insert(2)(Empty)))))))));;
let simpleBraun = (insert(5)(insert(12)(insert(7)(insert(0)(insert(3)(insert(2)(Empty)))))));;
(*

val simpleBraun = 

 Node (0, 
    Node (3, 
        Node (5, 
            Empty, 
            Empty), 
    Node (7, 
        Empty, 
        Empty)), 
    
    Node (2, 
        Node (12, 
            Empty, 
            Empty), 
    Empty))  

val braunHeapPQ : pq =  
Node (-3, 
    Node (0, 
        Node (2, 
            Node (12, Empty, Empty), Empty), 
        Node (99, Empty, Empty)),                                               
    Node (0, 
        Node (3, 
            Node (7, Empty, Empty), Empty), 
        Node (5, Empty, Empty))) 
   


*)