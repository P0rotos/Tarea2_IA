#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <string.h>
#include <chrono>
#include <random>

using namespace std;

int height_grid, width_grid, action_taken, action_taken2,current_episode;
int maxA[100][100], blocked[100][100];
float maxQ[100][100], cum_reward,Qvalues[100][100][4], reward[100][100],finalrw[50000];
int init_x_pos, init_y_pos, goalx, goaly, x_pos,y_pos, prev_x_pos, prev_y_pos, blockedx, blockedy,i,j,k;
ofstream reward_output;
ofstream forplot;

unsigned seed = chrono::system_clock::now().time_since_epoch().count();
default_random_engine gen1(seed);
uniform_real_distribution<float> distribution1(0.0, 1.0);
default_random_engine gen2(seed);
uniform_real_distribution<float> distribution2(0.0, 1.0);
default_random_engine gen3(seed);
uniform_int_distribution<int> distribution3(0, 3);

//////////////
//Setting value for learning parameters
int action_sel=2; // 1 is greedy, 2 is e-greedy
int environment= 2; // 1 is small grid, 2 is Cliff walking
int algorithm = 2; //1 is Q-learning, 2 is Sarsa
int stochastic_actions=1; // 0 is deterministic actions, 1 for stochastic actions
int num_episodes=3000; //total learning episodes
float learn_rate=0.1; // how much the agent weights each new sample
float disc_factor=0.99; // how much the agent weights future rewards
float exp_rate=0.05; // how much the agent explores
///////////////

void Initialize_environment(){
    if(environment==1){ 
        height_grid= 3;
        width_grid=4;
        goalx=3;
        goaly=2;
        init_x_pos=0;
        init_y_pos=0;
    }
    if(environment==2){
        height_grid= 4;
        width_grid=12;
        goalx=11;
        goaly=0;
        init_x_pos=0;
        init_y_pos=0;
    }
    for(i=0; i < width_grid; i++){
        for(j=0; j< height_grid; j++){
            if(environment==1){
                reward[i][j]=-0.04; //-1 if environment 2
                blocked[i][j]=0;
            }
            if(environment==2){
                reward[i][j]=-1;
                blocked[i][j]=0;
            }
            for(k=0; k<4; k++){
                Qvalues[i][j][k]=rand()%10;
                cout << "Initial Q value of cell [" <<i << ", " <<j << "] action " << k << " = " << Qvalues[i][j][k] << "\n";
            }
        }
    }
    if(environment==1){
        reward[goalx][goaly]=100;
        reward[goalx][(goaly-1)]=-100;
        blocked[1][1]=1;
    }
    if(environment==2){
        reward[goalx][goaly]=1;
        for(int h=1; h<goalx;h++){   
            reward[h][0]=-100;
        }
    }
}
int action_selection(){ 
    // Based on the action selection method chosen, it selects an action to execute next
    if(action_sel==1) {
        //Greedy, always selects the action with the largest Q value
        //return distribution3(gen3); //Currently returing a random action, need to code the greedy strategy
        int best_action = 0;
        float best_value = Qvalues[x_pos][y_pos][0];
        for (int i = 1; i < 4; i++){
            if (Qvalues[x_pos][y_pos][i] > best_value){
                best_value = Qvalues[x_pos][y_pos][i];
                best_action = i;
            }
        }
        return best_action;
    }
    if(action_sel==2){
        //epsilon-greedy, selects the action with the largest Q value with prob (1-exp_rate) and a random action with prob (exp_rate)
        //return rand()%4; //Currently returing a random action, need to code the e-greedy strategy
        if (distribution1(gen1) < exp_rate){
            return distribution3(gen3); 
        }else{
            int best_action = 0;
            float best_value = Qvalues[x_pos][y_pos][0];
            for (int i = 1; i < 4; i++){
                if (Qvalues[x_pos][y_pos][i] > best_value){
                    best_value = Qvalues[x_pos][y_pos][i];
                    best_action = i;
                }
            }
            return best_action;
        }
    }
    return 0;
}
void move(int action){
    prev_x_pos=x_pos; //Backup of the current position, which will become past position after this method
    prev_y_pos=y_pos;
    //Stochastic transition model (not known by the agent)
    //Assuming a .8 prob that the action will perform as intended, 0.1 prob. of moving instead to the right, 0.1 prob of moving instead to the left
    if(stochastic_actions){
        //Code here should change the value of variable action, based on the stochasticity of the action outcome
        float rnd_val = distribution2(gen2);
        if(rnd_val <= 0.1){
            action=(action+1)%4;
        }else if(0.9 <= rnd_val){
            action=(action-1)%4;
        }
    }
    action_taken2 = action;
    //After determining the real outcome of the chosen action, move the agent
    if(action==0){ // Up
        if((y_pos<(height_grid-1))&&(blocked[x_pos][y_pos+1]==0)){ //If there is no wall or obstacle Up from the agent
            y_pos=y_pos+1;  //move up
        }
    }
    if(action==1){  //Right
        if((x_pos<(width_grid-1))&&(blocked[x_pos+1][y_pos]==0)){ //If there is no wall or obstacle Right from the agent
            x_pos=x_pos+1; //Move right
        }
    }
    if(action==2){  //Down
        if((y_pos>0)&&(blocked[x_pos][y_pos-1]==0)){ //If there is no wall or obstacle Down from the agent
            y_pos=y_pos-1; // Move Down
        }
    }
    if(action==3){  //Left
        if((x_pos>0)&&(blocked[x_pos-1][y_pos]==0)){ //If there is no wall or obstacle Left from the agent
            x_pos=x_pos-1;//Move Left
        }
    }
}
void update_q_prev_state(){ //Updates the Q value of the previous state
    //Determine the max_a(Qvalue[x_pos][y_pos])
    float max_next_q = Qvalues[x_pos][y_pos][0];
    for (int a = 1; a < 4; a++) {
        if (Qvalues[x_pos][y_pos][a] > max_next_q) {
            max_next_q = Qvalues[x_pos][y_pos][a];
        }
    }
    //Update the Q value of the previous state and action if the agent has not reached a terminal state
    if(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))) ){
        Qvalues[prev_x_pos][prev_y_pos][action_taken2] = Qvalues[prev_x_pos][prev_y_pos][action_taken2] + learn_rate*(reward[x_pos][y_pos] + disc_factor*max_next_q - Qvalues[prev_x_pos][prev_y_pos][action_taken2]); //How should the Q values be updated?
    }
    else{//Update the Q value of the previous state and action if the agent has reached a terminal state
        Qvalues[prev_x_pos][prev_y_pos][action_taken2] += learn_rate * (reward[x_pos][y_pos] - Qvalues[prev_x_pos][prev_y_pos][action_taken2]);
    }
}
void update_q_prev_state_sarsa(){
    //Update the Q value of the previous state and action if the agent has not reached a terminal state
    if(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))     ) ){
       Qvalues[prev_x_pos][prev_y_pos][action_taken2] = Qvalues[prev_x_pos][prev_y_pos][action_taken2] + learn_rate*(reward[x_pos][y_pos] + disc_factor*Qvalues[x_pos][y_pos][action_taken] - Qvalues[prev_x_pos][prev_y_pos][action_taken2]);
    }
    else{//Update the Q value of the previous state and action if the agent has reached a terminal state
        Qvalues[prev_x_pos][prev_y_pos][action_taken2] += learn_rate * (reward[x_pos][y_pos] - Qvalues[prev_x_pos][prev_y_pos][action_taken2]);
    }
}
void Qlearning(){
   //Follow the  steps in the pseudocode in the slides
   move(action_selection());
   cum_reward=cum_reward+reward[x_pos][y_pos]; //Add the reward obtained by the agent to the cummulative reward of the agent in the current episode
}
void Sarsa(){
    move(action_taken);
    cum_reward=cum_reward+reward[x_pos][y_pos]; //Add the reward obtained by the agent to the cummulative reward of the agent in the current episode
    action_taken = action_selection();
}
void Multi_print_grid(){
    int x, y;
    for(y = (height_grid-1); y >=0 ; --y){
        for (x = 0; x < width_grid; ++x){
            if(blocked[x][y]==1) {
                cout << " \033[42m# \033[0m";
            }else{
                if ((x_pos==x)&&(y_pos==y)){
                    cout << " \033[44m1 \033[0m";
                }else{
                    cout << " \033[31m0 \033[0m";
                }
            }
        }
        printf("\n");
    }
}
int main(int argc, char* argv[]){
    srand(time(NULL));
    reward_output.open("Rewards.txt", ios_base::out);
    forplot.open("forplot.txt", ios_base::out);
    Initialize_environment();//Initialize the features of the chosen environment (goal and initial position, obstacles, rewards)
    for(i=0;i<num_episodes;i++){
        reward_output << "Episode " << i;
        current_episode=i;
        x_pos=init_x_pos;
        y_pos=init_y_pos;
        cum_reward=0;
        //If Sarsa was chosen as the algorithm:
        if(algorithm==2){
            action_taken= action_selection();
        }
        //While the agent has not reached a terminal state:
        while(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))     ) ){
            if(algorithm==1){
                Qlearning();
                update_q_prev_state();
            }
            if(algorithm==2){
                Sarsa();
                update_q_prev_state_sarsa();
            }
        }
        finalrw[i]=cum_reward;
        reward_output << " Total reward obtained: " <<finalrw[i] <<"\n";
        forplot << i << " " << finalrw[i] << "\n";
    }
    reward_output.close();
    forplot.close();
    return 0;
}