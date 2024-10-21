<template>
	<view class="text-box">
		<text class="text text1">{{ text }}</text>
		<br/>
		<text class="text text2">{{ comment }}</text>
		<br/>
		<text class="text text3">{{ sentiment }}</text>
	</view>
	<view>
		<button @tap="startRecord"><uni-icons type="mic-filled" size="20" color="green"></uni-icons> Start
			recording</button>
		<button @tap="endRecord"><uni-icons type="micoff-filled" size="20" color="red"></uni-icons> Stop
			recording</button>
		<button @tap="playVoice"><uni-icons type="sound-filled" size="20"></uni-icons> Play recording</button>
		<button @tap="uploadVoice"><uni-icons type="chatboxes" size="20"></uni-icons> Convert recording to text</button>
	</view>
</template>

<script>
	const recorderManager = uni.getRecorderManager();
	const innerAudioContext = uni.createInnerAudioContext();

	innerAudioContext.autoplay = true;

	export default {
		data() {
			return {
				text: 'Please click the Start recording button',
				comment: '',
				voicePath: '',
				sentiment: ''
			}
		},
		onLoad() {
			let self = this;
			recorderManager.onStop(function(res) {
				console.log('recorder stop' + JSON.stringify(res));
				self.voicePath = res.tempFilePath;
			});
		},
		methods: {
			startRecord() {
				console.log('Start recording');

				this.text = 'Start recording...';
			    this.comment = '';
				this.sentiment = '';

				recorderManager.start({
					format: 'wav'
				});
			},
			endRecord() {
				console.log('Stop recording');

				this.text = 'Stop recording...';
				this.comment = '';
				this.sentiment = '';

				recorderManager.stop();
			},
			playVoice() {
				console.log('Play recording');

				this.text = 'Play recording...';
				this.comment = '';
				this.sentiment = '';

				if (this.voicePath) {
					innerAudioContext.src = this.voicePath;
					innerAudioContext.play();
				}
			},
			uploadVoice() {
				console.log('Upload recording');
				console.log(this.voicePath);

				this.text = 'Upload recording...';
				this.comment = '';
				this.sentiment = '';

				if (this.voicePath) {

					const path = plus.io.convertLocalFileSystemURL(this.voicePath);
					console.log(path);

					this.task = uni.uploadFile({
						url: 'http://172.20.10.2:3000/uploadFile', // <--- please change your backend ip address here
						filePath: path,
						name: 'file',
						success: (uploadFileRes) => {
							console.log(uploadFileRes);
							let array = uploadFileRes.data.split("---")
							if(array.length === 3)
							{
								this.text = 'Recording content: ' + array[0];
								this.comment = array[1];
								this.sentiment = array[2];
							}
						},
						fail: (err) => {
							console.log(err)
						},
					})
				}
			}
		}
	}
</script>

<style>
	.text-box {
		padding: 20px 0;
		display: flex;
		min-height: 100px;
		background-color: #ffffff;
		justify-content: center;
		align-items: center;
		flex-wrap: wrap;
	}

	.text {
		margin-bottom: 20px;
		font-size: 15px;
		color: #353535;
		line-height: 27px;
		text-align: center;
	}
	
	.text2 {
		color: cadetblue;
		border-top: 1px dotted #353535;
		border-bottom: 1px dotted #353535;
	}
</style>