type 'a brt =
  Empty
| Node of 'a * 'a brt * 'a brt

let list1 = [1;3;4;5;6;8;9;2;3;4] ;;
(*


Convert To -> (1, )



*)

let rec braunTreeToList braunTree =
  let aux mergeNumbers treeLeft treeRight =
    match (treeLeft, treeRight)
    | (Node(rootL, ll, lr), Node(rootR, rl, rr)) -> rootL :: rootR  
  match braunTree with
  | Empty -> Empty
  | Node(root, treeLeft, treeRight) -> root :: 
  



    [root] @ braunTreeToList(treeLeft) @ braunTreeToList(treeLeft) 

let rec listToBraunTree = 
  match list with
  | hd::back -> Node(hd, listToBraunTree(), listToBraunTree())
  | [] -> Empty