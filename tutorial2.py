import pygame
from pong import Game
import os
import neat
import pickle
class PongGame:
    def __init__(self,window,width,height):
        self.game=Game(window,width,height)
        self.left_paddle=self.game.left_paddle
        self.right_paddle=self.game.right_paddle
        self.ball= self.game.ball

    def test_ai(self,genome,config):
        #we specified the keys for the left stricker- so it is for humanto play and right stricker is in control of ai . We can change it to both being ai by just simply copying what we did in train ai

        #net= neat.nn.FeedForwardNetwork.create(genome,config)   #activation for single player ai
        #activation for double player ai:
        net1 = neat.nn.FeedForwardNetwork.create(genome, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome, config)
        run=True
        clock=pygame.time.Clock()
        while run:
            #clock.tick(100)  # no of hit per sec


            #to stop the loop we will specify that when we hit the cross x on the window on which the game is played: stop the game. This can be done by using inbuilt pygame function : events
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    run=False # so this will be the last loop
                    break # to stop right here and not completing this last loop
            
            #we will create the specifications for the game like which keywill do what command:
            #to set the left slider move up by pressing W key on keyboard:

            #this code will make both ai play against each other in test ai
            output1 = net1.activate((self.left_paddle.y, self.ball.y, abs(self.left_paddle.x-self.ball.x )))
            decision1=output1.index(max(output1))
            if decision1==0:
                pass
            elif decision1==1:
                self.game.move_paddle(True,True)
            else:
                self.game.move_paddle(True,False)

            output2 = net2.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x-self.ball.x )))
            decision2=output2.index(max(output2))
            if decision2==0:
                pass
            elif decision2==1:
                self.game.move_paddle(False,True)
            else:
                self.game.move_paddle(False,False)
            
            '''#this code is for one player
            keys=pygame.key.get_pressed()
            if keys[pygame.K_w]: #we can do K_UP for using up key and down for down
                self.game.move_paddle(left=True,up=True) # this function first asks which slider to move- if left=True- command will be applied on left slidder, and up=True means it will move up when thiskey is pressed
            #now to specify that 'S' key will move it down :
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True,up=False)
            #game.loop() runs the game every time it is called and stores the info of what happened in that one cyle of pong. So to extract the information like points of both player which we will use further to specify the winning score and thus matching both players score to winning score to complete the game
            
            output = net.activate((self.right_paddle.y,self.ball.y,abs(self.right_paddle.x-self.ball.x)))
            decision = output.index(max(output))
            if decision==0:
                pass
            elif decision==1:
                self.game.move_paddle(False,True)
            else:
                self.game.move_paddle(False,False)'''

            game_info= self.game.loop() # one loop function will run game one time- so in endless while loop: endless time
             #to
            self.game.draw(True,False) # draw has two functions - hits and score, hits shows total hits by both player and score shows score of each player, by specifying true or false we  can change the setting.
            pygame.display.update()
        pygame.quit()


    def train_ai(self,genome1,genome2,config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        clock=pygame.time.Clock()
        while run:
            #clock.tick(1000)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            output1 = net1.activate((self.left_paddle.y, self.ball.y, abs(self.left_paddle.x-self.ball.x )))
            decision1=output1.index(max(output1))
            if decision1==0:
                pass
            elif decision1==1:
                self.game.move_paddle(True,True)
            else:
                self.game.move_paddle(True,False)

            output2 = net2.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x-self.ball.x )))
            decision2=output2.index(max(output2))
            if decision2==0:
                pass
            elif decision2==1:
                self.game.move_paddle(False,True)
            else:
                self.game.move_paddle(False,False)
            print(output1,output2)

            game_info = self.game.loop()


            self.game.draw(False,True)
            pygame.display.update()

            if game_info.left_score>=1 or game_info.right_score>=1 or game_info.left_hits>50 : # we do this to just immediately stop when ai or opponent misses the ball- so that it trains more rapidly because it can take a lot of time to hit the ball again
                self.calculate_fitness(genome1,genome2,game_info)
                break
    def calculate_fitness(self,genome1,genome2,game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits 
def eval_genomes(genomes,config):
    #we will train every sample neural nework or ai against all others. This is not a single player game in which env is fixed and machine can completely remember env to win, here it is playing against opponent who can do anything. So we will train the given ai model with all other so that each model until reaching fitness of 400 -max, see what all their opponents play and learn that ..so that they know the env now-which is what any of the other ai model plays
    width,height=700,500
    window=pygame.display.set_mode((width,height))
    for i,(genomesid1,genome1) in enumerate(genomes):
        if i==len(genomes)-1:
            break
        genome1.fitness=0
        for (genomesid2,genome2) in genomes[i+1:]:  #we use i+1 because before i= players or model have already played so no need to play again- they already know the fitness score against each other
            genome2.fitness=0 if genome2.fitness==None else genome2.fitness
            game=PongGame(window,width,height)
            game.train_ai(genome1,genome2,config)
            
def run_neat(config):
    #p=neat.Checkpointer.restore_checkpoint('neat-checkpoint-27')
    p=neat.Population(config) # we are creating neural network with the configuration fle info. 
    p.add_reporter(neat.StdOutReporter(True))
    stats= neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    winner= p.run(eval_genomes,50)
    with open("best.pickle","wb") as f:
        pickle.dump(winner,f)
def test_ai(config):
    width,height=700,500
    window=pygame.display.set_mode((width,height))
    with open("best.pickle","rb") as f:
        winner=pickle.load(f)
    game=PongGame(window,width,height)
    game.test_ai(winner,config)
if __name__=="__main__":
    local_dir=os.path.dirname(__file__)
    config_path=os.path.join(local_dir,"config.txt")
    config= neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)
     # we need to 
    #run_neat(config)
    test_ai(config)
    