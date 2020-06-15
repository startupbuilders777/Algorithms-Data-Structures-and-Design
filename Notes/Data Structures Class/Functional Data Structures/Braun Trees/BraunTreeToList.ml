type 'a brt =
  Empty
| Node of 'a * 'a brt * 'a brt

let list1 = [1;3;4;5;6;8;9;2;3;4] ;;
(*


Convert To -> (1, )



*)

let rec braunTreeToList braunTree =
  let rec createList root left right direction =
    match(left, right) with
    | (Empty, Empty) -> []
    | (Node(lRoot, ll, lr), Node(rRoot, rl, rr) ) when direction = "n" -> root :: createList root left right "l"
    | (Node(lRoot, ll, lr), Node(rRoot, rl, rr) ) when direction = "l" -> lRoot :: createList root left right "r"
    | (Node(lRoot, ll, lr), Node(rRoot, rl, rr) ) when direction = "r" -> rRoot :: createList root left right "x"
    |
  match braunTree with
  | Empty -> Empty
  | Node(root, treeLeft, treeRight) -> createList root treeLeft treeRight;;
  

(*

    [root] @ braunTreeToList(treeLeft) @ braunTreeToList(treeLeft) 

let rec listToBraunTree = 
  match list with
  | hd::back -> Node(hd, listToBraunTree(), listToBraunTree())
  | [] -> Empty

  *)