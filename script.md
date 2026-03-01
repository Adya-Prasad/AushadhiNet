"Hello! I'm Adya Prasad, and I'm presenting AushadhiNet-GATv2 model a Graph Neural Network model (trained from scratch) that predicts drug-drug interactions for cardiovascular disease treatment, and also for common treatments.

As per the World Heart Report 2025, Cardiovascular disease (CVD) remains the leading cause of death globally, with a projected 20.5 million deaths in 2025, rising to 35.6 million by 2050. And also the CVD treatments experiences the one of highest rate of adverse drug effect (ADR) primarily due to polypharmacy because heart patients typically require multiple simultaneous medications but ADRs are 53% of these ADR are potentially avoidable so here are some scope to make medication more robust.

ML models trained on massive amounts of therapeutically data may help physicians make more informed clinical decisions before prescribing drugs, there are few drug interaction checker but mostly work on rule-based rigidity: Cannot generalise to new drugs or unseen combination or few are very heavy to run.

[slide 5]  So for solution, I do some research and started by leveraging graph neural networks specifically graph attention network version 2, to solve this problem as a link prediction problem, considering previously developed DDI and addressing their achievement and shortcomings.Here how it works: "AushadhiNet solves this by generalizing molecular-level interaction patterns instead of just learning the drug pair. 

[slide 6] Let's understand its architecture in short, I formulate DDI prediction as a ‘link prediction problem’ on a drug interaction graph, extract three complementary views of each drug - Morgan fingerprints capture substructures, MACCS keys encode functional groups, and physicochemical descriptors quantify properties. Then properties goes to view Projection and Attention Fusion step where each molecular view is projected into a unified 384-dimensional embedding space through view-specific MLPs with BatchNorm and ReLU activations then their node embedding processed by a 4-layer GATv2 graph neural network with residual connections. This design addresses the vanishing gradient problem in deep GNNs by providing direct gradient pathways to early layers. Layer normalization after each GATv2 layer stabilizes activations and accelerates convergence. Then comes the edge Classification and Dual-Head Prediction where the input feature vector are fed into two separate Multi level perceptron (MLP) branches( say heads) the output vectors from both heads are concatenated which cause the prediction whether drugs interact AND the specific interaction mechanism and return the result

Let's see the actual code, the training pipeline. Here's my Jupyter notebook running on Google Colab.
1. First I installed the dependencies
2. then some memory optimization for faster training and dealing with memory constraints
3. Here is the main central configurations, it define the training hyper parameters and also some bool values (show with text selection), if you change the bool value, training componenents will change too, 
4. then I do loading data and data sanity check for quality training, I am using drug bank dataset and also hackathon csv data,  here you can see the data stats. data have Total 19 thousand plus drugs pair more than 17 hundered molecular structure, 
5. so here the actual pipeline begin, at this step, I extract the three multi-views of each drug, you can see the each view shape
6. the next comes the graph construction, and data splitting
7. I also add the data augmentation to avoid the overfitting and do better generalization
8. and here is the main graph neural network architecture, I'm not gonna deep init, 
9. I also created a trainig model tracker to save the best model
10. Here is another cruciel cell, the model training loop
11. and here i started the model trainig initialzation, I trained model for 200 epochs
12. and it is the training dashboard plot
12. (directly) let me show you the model training evaluation report, you can see the model accuracy, the model accuracy falls around 90% sometime ups or down depends upon colab gpu allocation
13. And below I am saving the metadata and best threshold to use during actual prediction
14. I also created a simple prediction function here in notebook, so you test the model youself on real data. You can see I test the model on three drugs it gives prediction on each drug pairs. The ui is quite ambigious so I also created a actual model inference using streamlit and coders workspace

let's see a quick deomonstration.
now I in local computer with save training weight and meta data,  (open streamlit tab, teraform and coder tab, do not show)
here is the quick look of my streamlit app script.
To run I started coder server and login and open the workspace panel, it handles the all deployment dependencies,  here you can see my drug interaction app workspace is running the streamlit.

I enable the patient profiling and adjust my info and entered the drugs name and click predict.
It predicted the each pair interaction and its ADR probability. interaction binary decision, 0 or 1 for no and yes and also probability of its binary decision, and then important interaction type and then its probability,

application has OCR feature, you can upload the prescription image, and it will extract the drugs name and do prediction on it, application pull the drug description from Pubchem API and also give personalized verdict
