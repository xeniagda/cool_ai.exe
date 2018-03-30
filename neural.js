function Neural(inputNodes, outputNodes, hiddenNodes, hiddenLayers, learningRate, batchSize, regularisation,id) {
	this.inputNodes = inputNodes;
	this.outputNodes = outputNodes;
	this.hiddenNodes = hiddenNodes;
	this.hiddenLayers = hiddenLayers;
	this.learningRate = learningRate;
	this.regularisation = regularisation;
	this.w = [];
	for (var i = 0; i < hiddenLayers + 1; i++)
		this.w[i] = [];
	this.b = [];
	for (var i = 0; i < hiddenLayers + 1; i++)
		this.b[i] = [];
	for (var i = 0; i < inputNodes; ++i) {
		this.w[0][i] = [];
		for (var j = 0; j < hiddenNodes; ++j)
			this.w[0][i][j] = Math.random() - 0.5;
	}
	for (var i = 1; i < hiddenLayers; i++)
		for (var j = 0; j < hiddenNodes; j++) {
			this.w[i][j] = [];
			for (var k = 0; k < hiddenNodes; k++)
				this.w[i][j][k] = Math.random() - 0.5;
		}
	for (var i = 0; i < hiddenNodes; ++i) {
		this.w[hiddenLayers][i] = [];
		for (var j = 0; j < outputNodes; ++j)
			this.w[this.w.length - 1][i][j] = Math.random() - 0.5;
	}
	for (var k = 0; k < hiddenLayers; k++)
		for (var i = 0; i < hiddenNodes; ++i)
			this.b[k][i] = Math.random() - 0.5;
	for (var i = 0; i < outputNodes; ++i)
		this.b[hiddenLayers][i] = Math.random() - 0.5;

	this.outputsHidden = [];
	this.inputs = [];
	this.outputs = [];
	this.deltasOutput = [];
	this.deltasHidden = [];
	this.numTests = 0;
	this.batchSize = batchSize;
	this.id = id;
}

Neural.prototype.next = function(input, target, strength) {
	//drawImageArray(input);
	var inputHidden = [
		[]
	];
	for (var i = 0; i < this.hiddenNodes; ++i) inputHidden[0][i] = 0;
	for (var i = 0; i < this.inputNodes; ++i)
		for (var j = 0; j < this.hiddenNodes; ++j)
			inputHidden[0][j] += this.w[0][i][j] * input[i];

	var outputHidden = [];
	for (var k = 0; k < this.hiddenLayers; ++k) {
		for (var j = 0; j < this.hiddenNodes; ++j)
			inputHidden[k][j] = inputHidden[k][j] + this.b[k][j];

		outputHidden[k] = inputHidden[k].map(x => sigmoid(x));

		if (k == this.hiddenLayers - 1) continue;

		inputHidden[k + 1] = [];
		for (var i = 0; i < this.hiddenNodes; ++i) inputHidden[k + 1][i] = 0;
		for (var i = 0; i < this.hiddenNodes; ++i)
			for (var j = 0; j < this.hiddenNodes; ++j)
				inputHidden[k + 1][j] += this.w[k + 1][i][j] * outputHidden[k][i];
	}

	var outputInput = [];
	for (var i = 0; i < this.outputNodes; ++i) outputInput[i] = 0;
	for (var i = 0; i < this.hiddenNodes; ++i)
		for (var j = 0; j < this.outputNodes; ++j)
			outputInput[j] += this.w[this.hiddenLayers][i][j] * outputHidden[this.hiddenLayers - 1][i];

	for (var j = 0; j < this.outputNodes; ++j)
		outputInput[j] = outputInput[j] + this.b[this.hiddenLayers][j];

	output = outputInput.map(x => sigmoid(x));

	var loss = 0;
	for (var i = 0; i < this.outputNodes; ++i)
		loss += (output[i] - target[i]) * (output[i] - target[i]);
	loss /= 2;

	var deltaOutput = [];
	for (var i = 0; i < this.outputNodes; i++)
		deltaOutput[i] = (output[i] - target[i]) //*sigmoidPrime(outputInput[i]);

	var deltaHidden = [];
	for (var k = this.hiddenLayers - 1; k >= 0; k--) {
		deltaHidden[k] = [];
		for (var i = 0; i < this.hiddenNodes; ++i) {
			var tmpSum = 0;
			if (k == this.hiddenLayers - 1)
				for (var j = 0; j < this.outputNodes; ++j)
					tmpSum += this.w[k + 1][i][j] * deltaOutput[j];
			else
				for (var j = 0; j < this.hiddenNodes; ++j)
					tmpSum += this.w[k + 1][i][j] * deltaHidden[k + 1][j];
			deltaHidden[k][i] = tmpSum * sigmoidPrime(inputHidden[k][i]);
		}
	}
	/*
	deltaK = [];
	for(var i = 0; i < this.outputNodes;++i)
		deltaK[i] = output[i]*(1 - output[i])*(output[i]-target[i]);

	deltaJ = [];
	for(var i = 0; i < this.hiddenNodes;++i){
		var tmpSum = 0;
		for(var j = 0; j < this.outputNodes;++j)
			tmpSum += deltaK[j]*this.w[this.hiddenLayers][i][j];
		deltaJ[i] = outputHidden[i]*(1 - outputHidden[i])*tmpSum;
	}*/
	if(!strength) strength = 1;
	for (var i = 0; i < strength; i++) {
		this.inputs.push(input);
		this.outputsHidden.push(outputHidden);
		this.deltasOutput.push(deltaOutput);
		this.deltasHidden.push(deltaHidden);
		this.numTests++;
		if (this.numTests == this.batchSize) {
			this.train();
			this.numTests = 0;
		}
	}
	return loss;
}

Neural.prototype.train = function() {
	for (var i = 0; i < this.inputNodes; ++i)
		for (var j = 0; j < this.hiddenNodes; ++j) {
			var tmpSum = 0;
			for (var x = 0; x < this.batchSize; x++)
				tmpSum += this.deltasHidden[x][0][j] * this.inputs[x][i];
			this.w[0][i][j] = this.w[0][i][j] * (1 - this.regularisation) - this.learningRate / this.batchSize * tmpSum;
		}

	for (var k = 1; k < this.hiddenLayers; ++k)
		for (var i = 0; i < this.hiddenNodes; ++i)
			for (var j = 0; j < this.hiddenNodes; ++j) {
				var tmpSum = 0;
				for (var x = 0; x < this.batchSize; x++)
					tmpSum += this.deltasHidden[x][k][j] * this.outputsHidden[x][k - 1][i];
				this.w[k][i][j] = this.w[k][i][j] * (1 - this.regularisation) - this.learningRate / this.batchSize * tmpSum;
			}

	for (var i = 0; i < this.hiddenNodes; ++i)
		for (var j = 0; j < this.outputNodes; ++j) {
			var tmpSum = 0;
			for (var x = 0; x < this.batchSize; x++)
				tmpSum += this.deltasOutput[x][j] * this.outputsHidden[x][this.hiddenLayers - 1][i];
			this.w[this.hiddenLayers][i][j] = this.w[this.hiddenLayers][i][j] * (1 - this.regularisation) - this.learningRate / this.batchSize * tmpSum;
		}

	for (var k = 1; k < this.hiddenLayers; ++k)
		for (var i = 0; i < this.hiddenNodes; ++i) {
			var tmpSum = 0;
			for (var x = 0; x < this.batchSize; x++)
				tmpSum += this.deltasHidden[x][k][i];
			this.b[k][i] += -this.learningRate / this.batchSize * tmpSum;
		}

	for (var i = 0; i < this.outputNodes; ++i) {
		var tmpSum = 0;
		for (var x = 0; x < this.batchSize; x++)
			tmpSum += this.deltasOutput[x][i];
		this.b[this.hiddenLayers][i] += -this.learningRate / this.batchSize * tmpSum;
	}

	/*for(var i = 0; i < this.inputNodes;++i)
		for(var j = 0; j < this.hiddenNodes;++j){
			this.w[0][i][j] += -this.learningRate*deltaJ[j]*input[i];
		}

	for(var i = 0; i < this.hiddenNodes;++i)
		for(var j = 0; j < this.outputNodes;++j){
			this.w[this.hiddenLayers][i][j] += -this.learningRate*deltaK[j]*outputHidden[i];
		}

	for(var i = 0; i < this.hiddenNodes;++i)
		this.b[i] += this.learningRate*deltaJ[i];*/

	this.inputs = [];
	this.outputsHidden = [];
	this.deltasOutput = [];
	this.deltasHidden = [];
}

Neural.prototype.test = function(input) {
	var inputHidden = [[]];
	for (var i = 0; i < this.hiddenNodes; ++i) inputHidden[0][i] = 0;
	for (var i = 0; i < this.inputNodes; ++i)
		for (var j = 0; j < this.hiddenNodes; ++j)
			inputHidden[0][j] += this.w[0][i][j] * input[i];

	var outputHidden = [];
	for (var k = 0; k < this.hiddenLayers; ++k) {
		for (var j = 0; j < this.hiddenNodes; ++j)
			inputHidden[k][j] = inputHidden[k][j] + this.b[k][j];

		outputHidden[k] = inputHidden[k].map(x => sigmoid(x));

		inputHidden[k + 1] = [];
		for (var i = 0; i < this.hiddenNodes; ++i) inputHidden[k + 1][i] = 0;

		if (k == this.hiddenLayers - 1) continue;
		for (var i = 0; i < this.hiddenNodes; ++i)
			for (var j = 0; j < this.hiddenNodes; ++j)
				inputHidden[k + 1][j] += this.w[k + 1][i][j] * outputHidden[k][i];
	}

	var outputInput = [];
	for (var i = 0; i < this.outputNodes; ++i) outputInput[i] = 0;
	for (var i = 0; i < this.hiddenNodes; ++i)
		for (var j = 0; j < this.outputNodes; ++j)
			outputInput[j] += this.w[this.hiddenLayers][i][j] * outputHidden[this.hiddenLayers - 1][i];

	for (var j = 0; j < this.outputNodes; ++j)
		outputInput[j] = outputInput[j] + this.b[this.hiddenLayers][j];

	var output = outputInput.map(x => sigmoid(x));
	return output;
}

Neural.prototype.save = function() {
	localStorage.setItem("w"+this.id,JSON.stringify(this.w));
	localStorage.setItem("b"+this.id,JSON.stringify(this.b));
}

Neural.prototype.load = function() {
	this.w = JSON.parse(localStorage.getItem("w"+this.id));
	this.b = JSON.parse(localStorage.getItem("b"+this.id));
}

Neural.prototype.saveFile = function() {
	$.post("../saveFile.php",{name:"w.json",content:JSON.stringify(this.w)});
	$.post("../saveFile.php",{name:"b.json",content:JSON.stringify(this.b)});
}

Neural.prototype.loadFile = function() {
	var self = this;
	$.get("w.json",function(res){
		self.w = (res);
	},"json");
	$.get("b.json",function(res){
		self.b = (res);
	},"json");
}

Neural.prototype.encodeW = function() {
	var size = 0;
	for (var i = 0; i < this.w.length; i++) 
		for (var j = 0; j < this.w[i].length; j++) 
			for (var k = 0; k < this.w[i][j].length; k++)
				++size;
	var wData = new Uint8Array(size*2);
	var index = 0;
	for (var i = 0; i < this.w.length; i++) 
		for (var j = 0; j < this.w[i].length; j++) 
			for (var k = 0; k < this.w[i][j].length; k++){
				wData[index*2] = (this.w[i][j][k]>0)?0:(1<<7);
				var num = Math.round(Math.abs(this.w[i][j][k])*(1<<12));
				for (var l = 0; l < 7; l++)
					wData[index*2] |= (((1<<(l+8))&num)>>8);
				for (var l = 0; l < 8; l++)
					wData[index*2+1] |= (((1<<l)&num));
				++index;
			}
	var base64 = btoa(
		wData.reduce((data, byte) => data + String.fromCharCode(byte), '')
	);
	return (base64);
}
Neural.prototype.encodeB = function() {
	var size = 0;
	for (var i = 0; i < this.b.length; i++) 
		for (var j = 0; j < this.b[i].length; j++) 
				++size;
	var bData = new Uint8Array(size*2);
	var index = 0;
	for (var i = 0; i < this.b.length; i++) 
		for (var j = 0; j < this.b[i].length; j++) {
				bData[index*2] = (this.b[i][j]>0)?0:(1<<7);
				var num = Math.round(Math.abs(this.b[i][j])*(1<<12));
				for (var l = 0; l < 7; l++)
					bData[index*2] |= (((1<<(l+8))&num)>>8);
				for (var l = 0; l < 8; l++)
					bData[index*2+1] |= (((1<<l)&num));
				++index;
			}
	var base64 = btoa(
		bData.reduce((data, byte) => data + String.fromCharCode(byte), '')
	);
	return (base64);
}

function sigmoid(x) {
	return 1 / (1 + Math.exp(-x));
}

function sigmoidPrime(z) {
	return sigmoid(z) * (1 - sigmoid(z));
}