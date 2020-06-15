(*braun heap*)
 
  type 'a brt =
    Empty
  | Node of 'a * 'a brt * 'a brt

let lst1 = [1;3;4;5;6;7;4;3;1;2;3];;

let rec insert (element: int) (bhpq)  =
      match bhpq with
      | Empty -> Node(element, Empty, Empty) 
      | Node (v, left, right) -> if (compare(element)(v)) = -1 then Node(element, insert(v)(right), left) 
                              else Node(v, insert(element)(right), left) ;;

let rec meld element leftTree rightTree = 
  match (leftTree, element, rightTree) with
  | (Empty, element, Empty) -> Node(element, Empty, Empty)
  | (Node(lval, ll, lr), element, Empty) -> if(element <= lval) then Node(element, meld(lval)(ll)(Empty), lr)
                                            else Node(lval, meld(element)(ll)(Empty), lr)

  | (Empty, element, Node(rval, rl, rr)) -> if(element <= rval) then Node(element, rl, meld(rval)(Empty)(rr))
                                            else Node(rval, rl, meld(element)(rr)(Empty))
  | (Node(lval, ll, lr), element, Node(rval, rl, rr)) -> 
    begin
      if (element <= rval && element <= lval) then Node(element, leftTree, rightTree)
      else if (rval <= element && rval <= lval) then Node(rval, leftTree, meld(element)(rl)(rr))
      else Node(lval, meld(element)(ll)(lr), rightTree)
  end;;

(*Should not be an empty tree*)
let rec deleteOne tree = 
  match tree with
  | Node(value, left, right) -> let (least, subtree) = deleteOne(left) in (least, Node(value, left, subtree))
  | Node(value, Empty, Empty) -> (value, Empty);;

let rec merge leftTree rightTree =
  let (element, tree) = deleteOne leftTree in
  meld(element)(rightTree)(tree);;

let rec heapify lst =
  let rec countNumberOfElements lst = 
    match lst with
    | hd::tl -> 1 + countNumberOfElements tl
    | [] -> 0 in
  let numberOfElements = countNumberOfElements(lst) in
  let rec splitList lst numberOfElements leftList = 
    match lst with 
    | hd::tl when numberOfElements > numberOfElements/2 -> splitList(tl)(numberOfElements - 1)(hd :: leftList)
    | _ -> (leftList, lst) in
  let isTree (element) =
    match element with 
    | Node(_,_,_) -> true
    | Empty -> true
    | _ -> false in
  let rec aux lstL lstR acc = 
    match (lstL, lstR) with
    | (a :: b :: tl, element :: rst) when isTree(a) = false -> aux(tl)(rst)( meld(Node(a, Empty, Empty))(element)(Node(b, Empty, Empty))  :: acc) 
    | (a :: b :: tl, element :: rst) -> aux(tl)(rst)( meld(a)(element)(b)  :: acc) 
    | (a :: [], _ as elements) -> merge(a)(aux(acc)(elements)([]))
    | ([], _ as elements) ->  aux(acc)(elements)([])
    | ([], []) -> 
        match acc with 
        | hd :: tl  -> merge(hd)(aux(lstL)(lstR)(tl))
        | hd :: [] -> hd
        | [] -> [] in
  let (left,elements) = splitList lst numberOfElements [] in
  aux left elements [] ;;

  