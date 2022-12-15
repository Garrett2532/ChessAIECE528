import chess
import RPi.GPIO as GPIO
import time
import board
import busio
import numpy
#import tensorflow as tf
from tensorflow import lite 
from digitalio import Direction, Pull
from adafruit_mcp230xx.mcp23017 import MCP23017


def lightBoard(animation,speed):                      #funtion to light the proper LED lights 
    sleepTime = .001
   # while(True): taking out inf loop until we get buttons working with code
    for frame in range(len(animation)):                                     #3d loop containng frames in the animation and lines in the frams
                for pause in range (speed):
                    for i in range(8):
                          GPIO.output(cathodes[0],animation[frame][i][0])   #setting the cathode to value in the frame
                          GPIO.output(cathodes[1],animation[frame][i][1])
                          GPIO.output(cathodes[2],animation[frame][i][2])
                          GPIO.output(cathodes[3],animation[frame][i][3])
                          GPIO.output(cathodes[4],animation[frame][i][4])
                          GPIO.output(cathodes[5],animation[frame][i][5])    
                          GPIO.output(cathodes[6],animation[frame][i][6])                    
                          GPIO.output(cathodes[7],animation[frame][i][7])

                          GPIO.output(anodes[i],1)                          #set anode of that node to 1
                          time.sleep(sleepTime)                             # wait to keep lit for a while
                          GPIO.output(anodes[i],0)

def getLegalMoves(chessboard,s):


    cfile = s[0]                           #file is the A in A1 (0-7) to be done with buttons later 
    rank = s[1]                                       #rank is the 1 in A1 (0-7)

    spot  = chess.square(cfile, rank)
    legalMoves = (list(chessboard.generate_legal_moves(from_mask=chess.BB_SQUARES[spot])))
    legalIndex = []
    for M in legalMoves:
        t = chessboard.san(M)
        if(len(t) > 2):
            if(t[-1] == '+' or t[-1] == '#'):
                t = t[:-1]
        while(len(t) > 2):                          #get rid of any symbols and like R for rook or + for check or x for taking a peice
         t= t[1:]
        legalIndex.append(chess.parse_square(t))


    boardlight  =  [[1,1,1,1,1,1,1,1],              #get inital board state for LEDs
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1]]

    for i in legalIndex:
         r = int(i/8)
         index = i%8                                 #get rank and file for each peice in legal index and convert to a numerical index and change index of led baord to be lit on those spots
         boardlight[r][index] = 0

    return boardlight





def getSquare(chessboard,toFrom,ani):
    #code for the buttons will go here
    #GPIO.setmode(GPIO.BCM) 
    #GPIO.setmode(GPIO.BOARD)
    #GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_UP)#USED for GPIO.BCM
    GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    i2c = busio.I2C(board.SCL, board.SDA)

# Create an instance of either the MCP23008 or MCP23017 class depending on
# which chip you're using:
#mcp = MCP23008(i2c)  # MCP23008
    mcp = MCP23017(i2c)  # MCP23017
    GPIO.setmode(GPIO.BCM)
    rank = []
    for pin in range(0,8):
        rank.append(mcp.get_pin(pin))
    File = []
    for pin in range(8,16):
        File.append(mcp.get_pin(pin))

    for pin in rank:
        pin.direction = Direction.INPUT
        pin.pull = Pull.UP
    
    for pin in File:
        pin.direction = Direction.INPUT
        pin.pull = Pull.UP
    rankb = False
    fileb = False
    rankval = -1
    fileval = -1
    while(True):
        if rankb == True:
            break
        for num, button in enumerate(rank):
            if(toFrom):
                lightBoard(ani,minTime)#This line givng BCM error 
            if not button.value:
                print("Button ", num , "pressed")
                rankval = num
                rankb = True
                break
            else:
                state = GPIO.input(21);
                if(state==False):
                   startTime = time.time()
                   while(GPIO.input(21) == False):
                       if(time.time()-startTime > 3):
                       #restart the game
                           print("restarting game...")
                           print("")
                           chessboard.reset()
                           chessboard.clear_stack()
                           lightBoard([reset],100)
                           print(chessboard)
                           return(-2)
                   if(toFrom):
                       return(-1)
    
    while(True):
        if fileb ==  True:
            break
        for num,button in enumerate(File):
            if(toFrom):
                lightBoard(ani,1)
            if not button.value :
                print( "Button ", num ," pressed")
                print("")
                fileval = num
                fileb = True
                break
    
    return[rankval,fileval]                 #first posit is the file, A in A1 will be numbered 0-7
                                #second position is the rank, 1 in A1 and will be numberered 0-7

def rotateMatrix(mat):

    if not mat or not len(mat):
        return

    N = len(mat)

    for i in range(N // 2):
        for j in range(N):
            temp = mat[i][j]
            mat[i][j] = mat[N - i -1][N -j -1]
            mat[N - i - 1][N- j - 1] = temp
    if N %2 ==1:
        for j in range(N // 2):
            temp = mat[N //2][j]
            mat[N //2][j] = mat[N // 2][N -j -1]
            mat[N //2][N-j -1] = temp


def gameOver(baord):
    if(baord.is_checkmate()):
            return True         #returns true if in a checkmate state need to be updated to check for stalemate or good peices 
    else:
            return False
def getAni(i):
    lity =    [[1,1,1,1,1,1,1,1],              #get inital board state for LEDs
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1]]
      
    r = int(i/8)
    index = i%8
    lity[r][index] = 0

    return lity
########Added AI stuff starting here
squares_index = {
  'a': 0,
  'b': 1,
  'c': 2,
  'd': 3,
  'e': 4,
  'f': 5,
  'g': 6,
  'h': 7
}


# example: h3 -> 17
def square_to_index(square):
  letter = chess.square_name(square)
  return 8 - int(letter[1]), squares_index[letter[0]]

def letter_to_index(letter):
    return 8 - int(letter[1]), squares_index[letter[0]]

def split_dims(board):
  # this is the 3d matrix
  board3d = numpy.zeros((14, 8, 8), dtype=numpy.int8)

  # here we add the pieces's view on the matrix
  for piece in chess.PIECE_TYPES:
    for square in board.pieces(piece, chess.WHITE):
      idx = numpy.unravel_index(square, (8, 8))
      board3d[piece - 1][7 - idx[0]][idx[1]] = 1
    for square in board.pieces(piece, chess.BLACK):
      idx = numpy.unravel_index(square, (8, 8))
      board3d[piece + 5][7 - idx[0]][idx[1]] = 1

  # add attacks and valid moves too
  # so the network knows what is being attacked
  aux = board.turn
  board.turn = chess.WHITE
  for move in board.legal_moves:
      i, j = square_to_index(move.to_square)
      board3d[12][i][j] = 1
  board.turn = chess.BLACK
  for move in board.legal_moves:
      i, j = square_to_index(move.to_square)
      board3d[13][i][j] = 1
  board.turn = aux

  return board3d


def minimax_eval(board):
  board3d = split_dims(board)
  board3d = numpy.expand_dims(board3d, 0)
  board2 = board3d.reshape((1,14,8,8))
  board2 = numpy.float32(board2)
  interpreter.set_tensor(input_index, board2)
  interpreter.invoke()
  prediction = interpreter.get_tensor(output_index)
  fpred = prediction[0][0]
  return fpred

def minimax(board, depth, alpha, beta, maximizing_player):
  if depth == 0 or board.is_game_over():
    return minimax_eval(board)
  
  if maximizing_player:
    max_eval = -numpy.inf
    for move in board.legal_moves:
      board.push(move)
      eval = minimax(board, depth - 1, alpha, beta, False)
      board.pop()
      max_eval = max(max_eval, eval)
      alpha = max(alpha, eval)
      if beta <= alpha:
        break
    return max_eval
  else:
    min_eval = numpy.inf
    for move in board.legal_moves:
      board.push(move)
      eval = minimax(board, depth - 1, alpha, beta, True)
      board.pop()
      min_eval = min(min_eval, eval)
      beta = min(beta, eval)
      if beta <= alpha:
        break
    return min_eval


# this is the actual function that gets the move from the neural network
def get_ai_move(board,depth):
  max_move = None
  max_eval = -numpy.inf

  for move in board.legal_moves:
    board.push(move)
    eval = minimax(board, depth - 1, -numpy.inf, numpy.inf, False)
    board.pop()
    if eval > max_eval:
      max_eval = eval
      max_move = move
  
  return max_move


def ai_to_move(chessboard, depth):
    move = get_ai_move(chessboard, depth)
    move2 = str(move)
    fsq = move2[:2]
    tsq = move2[2:]
    x = 7-int (letterToNumber[fsq[0]])
    y = 8-int (fsq[1])
    ani = [getAni(8*y + x)]
    while(True):
        AIFrom = getSquare(chessboard, True, ani)
        AIFromy = AIFrom[0]
        AIFromx = AIFrom[1]
        f = buttonxToFile[AIFrom[0]]
        rank = 8- int(AIFrom[1])
        if(str(f)+str(rank) == fsq) :
            #print('Done')
            break
        else:
            print('That is not the square that the AI selected, try again\n')
            lightBoard([badmove], midTime)
    x = 7-int (letterToNumber[tsq[0]])
    y = 8-int (tsq[1])
    ani = [getAni(8*y + x)]
    while(True):
        AIFrom = getSquare(chessboard, True, ani)
        AIFromy = AIFrom[0]
        AIFromx = AIFrom[1]
        f = buttonxToFile[AIFrom[0]]
        rank = 8- int(AIFrom[1])
        if(str(f) + str(rank)==tsq):
            #print('Done2')
            break
        else:
            print('That is not the square the AI selected, try again\n')
            lightBoard([badmove],midTime)
    for i in range(2):
        lightBoard([dim], midTime)
        lightBoard(ani, midTime)
    chessboard.push(move)
    return chessboard.is_checkmate()

def player_to_move(chessboard):
    while(True):
        foundLegalMove = False
        fromsq = getSquare(chessboard, False, None)
        if(fromsq == -2):
            break
        lightSquare = chess.square(fromsq[0] , fromsq[1])
        ani = [getAni(lightSquare)]
        lightBoard(ani, midTime)
        fsquare = chess.square(7-fromsq[0],7- fromsq[1])
        legalMoves = list(chessboard.generate_legal_moves(from_mask=chess.BB_SQUARES[fsquare]))
        legalSquares = []
        for move in legalMoves:
            longMove = chessboard.san(move)
            shortMove = ''
            if longMove == 'O-O-O':
                if(chessboard.turn):#white queenside
                    legalSquares.append('c1')
                else:
                    legalSquares.append('c8')#black queenside
                break
            if longMove == 'O-O':
                if(chessboard.turn):#white kingsside
                    legalSquares.append('g1')
                else:
                    legalSquares.append('g8')#black kingsside
                break
            for c in longMove:
                if c in legalChars:
                    shortMove += c
            legalSquares.append(shortMove)
        if(len(legalMoves) == 0):
            lightBoard([badmove],midTime)
            print("This square has no legal moves \n")
            tosq = 1
        else:
            while(True):
                ani = getLegalSquaresLit(legalSquares)
                tosq = getSquare(chessboard, True, [ani])
                if(tosq == -1 ):
                    print('resetnig move \n')
                    break
                elif( tosq == -2):
                    fromsq = -2
                    break
                x = 7-tosq[0]
                y = 8-tosq[1]
                print(legalMoves)
                for move in legalMoves:
                    longMove = chessboard.san(move)
                    shortMove = ''
                    if longMove == 'O-O-O':
                        print('gotHere qsc')
                        if(chessboard.turn):#white queenside
                            shortMove=('c1')
                        else:
                            shortMove=('c8')#black queenside
                    if longMove == 'O-O':
                        print('gotHere ksc')
                        if(chessboard.turn):#white kingsside
                            shortMove=('g1')
                        else:
                            shortMove=('g8')#black kingsside
                    for c in longMove:
                        if c in legalChars:
                            shortMove += c
                    rank = numberToLetter[x]
                    if len(shortMove) > 2:
                        shortMove = shortMove[-2:]
                    print('shortMove ' + shortMove)
                    print('Rank + file '+ rank + str(y))
                    print('sm == rank+y ' +str(shortMove == rank +str(y)))
                    if(shortMove == (rank + str(y))):
                        chessboard.push(move)
                        foundLegalMove = True
                        movesq = shortMove
                        break
                if foundLegalMove:
                    ani = [getLegalSquaresLit([movesq])]
                    rotateMatrix(ani)
                    for i in range (2):
                        lightBoard([dim], midTime)
                        lightBoard(ani, midTime)
                    return chessboard.is_checkmate()
                else:
                    print('That peice cannot move like that, try again or reset the move')
                    lightBoard([badmove], midTime)
                    tosq = 1
        if(tosq == -1):
            print('retryMove')
        elif(tosq == -2):
            print('Forfit by player')
            return -2         

def getLegalSquaresLit(legalSquares):
    dim =       [[1,1,1,1,1,1,1,1],              
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1]]
    print(legalSquares)
    for square in legalSquares:
        if len(square) > 2:
            square = square[-2:]
        rank =7-letterToNumber[square[0]]
        f = 7-int(square[1])+1
        dim[f][rank] = 0
    return dim               



def startGame(depth):
    lightBoard([reset], longestTime)
    forit = False
    print(chessboard)
    print("")
    isCheckmate = False
    while not isCheckmate:
        isCheckmate = ai_to_move(chessboard, depth)#number represents the depth of the search 
        if not(isCheckmate):
            print(chessboard)
            print("")
            isCheckmate = player_to_move(chessboard)
            if isCheckmate == -2:
                print('Oof a forfit')
                forfit = True
                isCheckmate = True
            print(chessboard)
            print("")
        else:
            break
    winner = 'White'
    if((chessboard.turn) and (forit == True)):
        winner = 'Black'
    print('Game over, ' + winner + ' won')
    print('Starting new game...\n')
    startGame(depth)

############################PROGRAM STARTS HERE######################

if __name__ == '__main__':


    cathodes = [4,17,27,22,10,9,11,0]               #these values are used for GPIO.BCM not GPIO.board
    anodes = [14,15,18,23,24,25,8,7]
    GPIO.setmode(GPIO.BCM)
    for cathode in cathodes:
         GPIO.setup(cathode, GPIO.OUT)
         GPIO.output(cathode, 0)
    for anode in anodes:
         GPIO.setup(anode, GPIO.OUT)
         GPIO.output(anode, 0)

    reset =         [[0,0,0,0,0,0,0,0],              #get inital board state for LEDs
                [0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0],              #get inital board state for LEDs
                [0,0,0,0,0,0,0,0]]

    dim =          [[1,1,1,1,1,1,1,1],              #get inital board state for LEDs
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1]]

    badmove =      [[0,1,0,1,0,1,0,1],              #get inital board state for LEDs
                [1,0,1,0,1,0,1,0],
                [0,1,0,1,0,1,0,1],
                [1,0,1,0,1,0,1,0],
                [0,1,0,1,0,1,0,1],
                [1,0,1,0,1,0,1,0],
                [0,1,0,1,0,1,0,1],
                [1,0,1,0,1,0,1,0]]
    minTime = 5#5
    shortTime =50 #50
    midTime = 75#75
    longTime = 100#100
    longestTime =200 #200

    letterToNumber = {'a': 0, 'b': 1, 'c': 2, 'd':3 , 'e': 4, 'f': 5, 'g': 6, 'h':7}
    numberToLetter = {0: 'a', 1: 'b', 2:'c', 3:'d', 4:'e', 5:'f', 6: 'g', 7: 'h'}
    buttonxToFile = {0: 'h', 1: 'g', 2: 'f', 3: 'e', 4: 'd', 5: 'c', 6: 'b', 7: 'a'}
    legalChars = ['a','b','c','d','e','f','g','h','1','2','3','4','5','6','7','8']



    #cathodes = [7,11, 13, 15, 19, 21, 23, 27]        #These are used for GPIO.BOARD
    #anodes = [8, 10, 12, 16, 18, 22, 24, 26]
    chessboard = chess.Board()

    final_model_file = './final.tflite' #change this if using a different model
    #creat an interpretor and test 
    interpreter = lite.Interpreter(model_path=str('final.tflite'))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    startGame(2)
