<h1>Heart Failure Prediction — Neural Networks (HW2)</h1>

<p>
This project extends the HF pipeline with a <strong>PyTorch</strong> classifier. We implement core activations
from scratch, verify a single-layer forward pass, and train a compact MLP on the SVMLight features
produced in HW1. Below are the exact settings and observed results from the finalized run.
</p>

<hr/>

<h2>📁 Project Layout</h2>
<pre><code>.
├─ HW2_NN.ipynb
├─ HW2_NN-lib/
│  └─ data/
│     ├─ features_svmlight.train
│     └─ features_svmlight.val
└─ README.md
</code></pre>

<hr/>

<h2>🧠 Task Summary</h2>
<ol>
  <li><strong>From-scratch activations</strong>: <code>relu</code>, <code>sigmoid</code>, <code>softmax</code> implemented with primitive ops (no torch activations).</li>
  <li><strong>Single-layer network</strong>: compute <code>sigmoid(X W^T + b)</code>; passes numeric unit tests.</li>
  <li><strong>MLP model</strong>: 1473 → 64 → 32 → Dropout(0.5) → 1 with ReLU, ReLU, Sigmoid.</li>
  <li><strong>Training</strong>: <code>BCELoss</code> + <code>SGD(lr=0.01)</code>, batch size 32, 100 epochs, periodic validation.</li>
  <li><strong>Evaluation</strong>: threshold probs at 0.5 → classes; report Acc, AUC, Precision, Recall, F1.</li>
</ol>

<hr/>

<h2>📦 Data &amp; Loaders</h2>
<ul>
  <li><code>X_train</code> shape: <code>[2485, 1473]</code>; <code>Y_train</code> shape: <code>[2485]</code></li>
  <li><code>X_val</code> shape: <code>[604, 1473]</code>; <code>Y_val</code> shape: <code>[604]</code></li>
  <li><code>batch_size = 32</code>; train batches: <code>78</code>; val batches: <code>19</code>; train loader shuffles.</li>
</ul>

<hr/>

<h2>🧩 Key Implementations</h2>

<h3>1) Activations (from scratch)</h3>
<ul>
  <li><strong>ReLU</strong>: <code>x * (x &gt; 0)</code> (keeps autograd graph).</li>
  <li><strong>Sigmoid</strong>: <code>1 / (1 + exp(-x))</code>.</li>
  <li><strong>Softmax</strong> (row-wise, stable): subtract row max → <code>exp</code> → row-normalize.</li>
</ul>

<h3>2) Single-layer forward</h3>
<p>
<code>lin = X @ W^T + b</code> (broadcast bias), then apply your <code>sigmoid</code>. Verified against provided test cases.
</p>

<h3>3) Network architecture</h3>
<pre><code># Module names kept as required for autograder checks:
fc1: Linear(1473 → 64)
fc2: Linear(64 → 32)
dropout: Dropout(p=0.5)
fc3: Linear(32 → 1)

forward: ReLU(fc1) → ReLU(fc2) → Dropout → Sigmoid(fc3)  →  output shape: (B, 1)
</code></pre>

<h3>4) Loss &amp; Optimizer</h3>
<ul>
  <li><strong>Loss</strong>: <code>nn.BCELoss()</code> (unit check with sample tensors ≈ <code>1.3391</code>).</li>
  <li><strong>Optimizer</strong>: <code>torch.optim.SGD(model.parameters(), lr=0.01)</code>.</li>
</ul>

<h3>5) Training loop</h3>
<ol>
  <li><code>optimizer.zero_grad()</code></li>
  <li><code>y_hat = model(x)</code></li>
  <li><code>loss = BCELoss(y_hat, y.view(-1,1))</code>  <!-- target/pred shape match --></li>
  <li><code>loss.backward()</code></li>
  <li><code>optimizer.step()</code></li>
</ol>
<p>Logged mean training loss per epoch and evaluated on the validation set every 20 epochs.</p>

<hr/>

<h2>🚦 Results</h2>

<h3>Before training (random init)</h3>
<ul>
  <li><strong>Train</strong>: acc ≈ <code>0.479</code>, auc ≈ <code>0.482</code>, precision ≈ <code>0.567</code>, recall ≈ <code>0.453</code>, f1 ≈ <code>0.504</code></li>
  <li><strong>Val</strong>:   acc ≈ <code>0.450</code>, auc ≈ <code>0.474</code>, precision ≈ <code>0.583</code>, recall ≈ <code>0.335</code>, f1 ≈ <code>0.426</code></li>
</ul>

<h3>Validation metrics during training</h3>
<table>
  <thead>
    <tr><th>Epoch</th><th>Acc</th><th>AUC</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>0.608</td><td>0.495</td><td>0.608</td><td>1.000</td><td>0.756</td></tr>
    <tr><td>20</td><td>0.608</td><td>0.677</td><td>0.608</td><td>1.000</td><td>0.756</td></tr>
    <tr><td>40</td><td>0.664</td><td>0.704</td><td>0.644</td><td>0.997</td><td>0.783</td></tr>
    <tr><td>60</td><td><strong>0.725</strong></td><td>0.732</td><td>0.721</td><td>0.894</td><td><strong>0.798</strong></td></tr>
    <tr><td>80</td><td>0.717</td><td><strong>0.741</strong></td><td>0.736</td><td>0.834</td><td>0.782</td></tr>
  </tbody>
</table>

<p>
<strong>Highlights:</strong> Best <em>F1</em> observed at epoch 60 (≈ <code>0.798</code>) with accuracy ≈ <code>0.725</code>. Best <em>AUC</em> observed at epoch 80 (≈ <code>0.741</code>).
Training loss decreased monotonically at checkpoints (0→20→40→60→80), satisfying the assignment’s convergence check.
</p>

<hr/>

<h2>📊 Reproducibility</h2>
<ul>
  <li>Seeds: <code>seed = 24</code> for <code>random</code>, <code>numpy</code>, <code>torch</code>; also <code>PYTHONHASHSEED</code> set to <code>"24"</code>.</li>
  <li>Use <code>model.eval()</code> for evaluation to disable Dropout.</li>
  <li>Keep training DataLoader with <code>shuffle=True</code>; validation without shuffle.</li>
</ul>

<hr/>

<h2>⚠️ Challenges &amp; Solutions</h2>
<ol>
  <li>
    <strong>“From-scratch” enforcement for activations</strong><br/>
    <em>Issue:</em> Autograder deletes <code>torch.relu/sigmoid/softmax</code> to catch wrappers.<br/>
    <em>Solution:</em> Implemented each using primitive ops (<code>exp</code>, comparisons, row-wise normalization) and a numerically stable Softmax (subtract row max).
  </li>
  <li>
    <strong>Single-layer shape mismatch</strong><br/>
    <em>Issue:</em> Incorrect matrix multiply orientation (<code>X @ W</code>) or missing bias broadcast causes wrong shape/value.<br/>
    <em>Solution:</em> Use <code>X @ W.T + b</code>, then apply custom <code>sigmoid</code>; verified against provided numeric tests.
  </li>
  <li>
    <strong>BCELoss target-size error</strong><br/>
    <em>Issue:</em> <code>y_hat</code> is <code>(B,1)</code>, labels are <code>(B,)</code> → size mismatch.<br/>
    <em>Solution:</em> <code>loss = criterion(y_hat, y.view(-1,1))</code> to align shapes.
  </li>
  <li>
    <strong>Wrong AUC / metric crashes</strong><br/>
    <em>Issue:</em> Passing class labels to AUC or not thresholding for precision/recall/F1.<br/>
    <em>Solution:</em> Keep probabilities for AUC; compute <code>y_pred = (y_hat &gt; 0.5)</code> for class metrics inside <code>evaluate()</code>.
  </li>
  <li>
    <strong>Convergence / loss not decreasing</strong><br/>
    <em>Issue:</em> Missing <code>optimizer.zero_grad()</code> or learning-rate sensitivity.<br/>
    <em>Solution:</em> Follow the 5-step loop strictly; with <code>lr=0.01</code> the model converged and passed the monotonic loss check.
  </li>
</ol>

<hr/>

<h2>🛠️ Environment</h2>
<pre><code>python &gt;= 3.9
pip install -r requirements.txt
</code></pre>

<p><strong>requirements.txt</strong></p>
<pre><code>torch
numpy
scikit-learn
scipy
</code></pre>

<hr/>

<h2>🔌 Handy Snippet</h2>
<pre><code>from sklearn.datasets import load_svmlight_file
Xtr, ytr = load_svmlight_file("HW2_NN-lib/data/features_svmlight.train")
Xva, yva = load_svmlight_file("HW2_NN-lib/data/features_svmlight.val")
</code></pre>

<hr/>

<h2>📄 License</h2>
<p>MIT.</p>

<hr/>

<h2>🙌 Acknowledgements</h2>
<p>Course staff for the baseline utilities, metrics helpers, and assignment scaffolding.</p>
