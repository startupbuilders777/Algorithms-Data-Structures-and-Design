'''

110.00,150.00
None
17.99
24.00,25.00,25.00
'''


import bisect

def sortedHotelPrices():
  N = int(input())
  
  queries = []
  
  suppliersMap = {}
  
  for i in range(N):
    ins = input()
    ins = ins.split(',')
    
    if(suppliersMap.get(ins[0]) is None):
      suppliersMap[ins[0]] = [ {"supplier": ins[1], "price": float(ins[2])} ]
    else:
      suppliersMap[ins[0]].append( {"supplier": ins[1], "price": float(ins[2])} )
  
  
  for i in range(int(input())):
    ins = input()
    ins = ins.split(',')
    city = ins[0]
    days = int(ins[1])
    
    vendors = suppliersMap[city]
    
    price = []
    
    for vendor in vendors:
      supplier = vendor["supplier"]
      vendorPrice = vendor["price"]
      
      if(supplier == 'A' and days == 1):
        #DOES INSERT SORT SO ITS INSERTED IN DESCENDING ORDER
        bisect.insort(price, vendorPrice * 1.5) 
      elif(supplier == 'B' and days <= 3):
        price.append(None)
      elif(supplier == 'C' and days >= 7):
        bisect.insort(price, vendorPrice * 0.9)
      elif(supplier == 'D' and days < 7):
        bisect.insort(price, vendorPrice * 1.1)
      else:
        bisect.insort(price, vendorPrice)

    formattedPrices = []
    for i in price:
      if i is not None: 
        formattedPrices.append(str("{0:.2f}".format(i)) )
      else:
        formattedPrices.append("None")
    
    print(",".join(formattedPrices)) 
    
  
sortedHotelPrices()