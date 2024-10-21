import { Controller, Get, Post, UploadedFile, UseInterceptors } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { Express } from 'express';
import { AppService } from './app.service';
import { promises as fsPromises } from 'fs';
import * as path from 'path';
import * as child from 'child_process';
import { stderr } from 'process';
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Get()
  getHello(): string {
    return this.appService.getHello();
  }

  @Post('uploadFile')
  @UseInterceptors(FileInterceptor('file'))
  async uploadFile(@UploadedFile() file: Express.Multer.File): Promise<string> {
    // Handle the uploaded file here
    // const voiceFile = './files/output_file.wav';
    // const fileExists = await checkFileExists(voiceFile);

    // if(fileExists)
    // {
    //   await deleteFile(voiceFile);
    // }
    
    // const destinationFolder = './files';
    // const filePath = path.join(destinationFolder, file.originalname);
    // await fsPromises.writeFile(filePath, file.buffer);
    // const command = `ffmpeg -i ${ destinationFolder }/${ file.originalname } ${ destinationFolder }/output_file.wav`;

    // const result = await new Promise((resolve, reject) => {
    //   child.exec(command, (error, stdout, stderr) => {
    //     if(error)
    //     {
    //       console.error(error);
    //       reject(error);
    //     }
  
    //     console.log(stdout);
    //     resolve(runPythonScript());
    //   })
    // });
    

    // try {
    //   console.log('File saved successfully:', filePath);
    //   return 'File uploaded and saved successfully and voice is ' + result;
    // } catch (error) {
    //   console.error('Error saving file:', error);
    //   throw new Error('Failed to save file');
    // }

    return runPythonScript();
  }
}

async function deleteFile(filePath: string): Promise<void> {
  try {
    await fsPromises.unlink(filePath);
    console.log('File deleted successfully:', filePath);
  } catch (error) {
    console.error('Error delete file:', error);
    throw new Error('Error delete file');
  }
}

async function checkFileExists(filePath: string): Promise<boolean> {
  try {
    await fsPromises.access(filePath);
    return true;
  } catch (error) {
    return false;
  }
}

// Run first Python script
async function runFirstScript() {
  try {
    const { stdout, stderr } = await execPromise('python Py/WeNet.py');
    // if (stderr) {
    //   console.error(`Error in script1: ${stderr}`);
    // }
    let stdoutR = stdout.replace('Module "torch_npu" not found. "pip install torch_npu"                 if you are using Ascend NPU, otherwise, ignore it\r\n', '')
    .replace(/�b/g,' ').toLowerCase().trim();
    console.log(`First script output: ${stdoutR}`);
    return stdoutR;
  } catch (error) {
    console.error(`Failed to run script1: ${error}`);
    throw error;
  }
}

async function runThirdScript(firstScriptResult) {
  try {
    const { stdout, stderr } = await execPromise(`python Py/Sentiment.py --mode test --sentence "${firstScriptResult}"`);
    // if (stderr) {
    //   console.error(`Error in script3: ${stderr}`);
    // }
    console.log(`Third script output: ${stdout}`);
    return stdout.trim();
  } catch (error) {
    console.error(`Failed to run script3: ${error}`);
    throw error;
  }
}

async function runSecondScript() {
  try {
    const { stdout, stderr } = await execPromise('python Py/SoundClassification.py --mode test --model-path Py/sound_classification_model.keras --test-file files/output_file.wav');
    // if (stderr) {
    //   console.error(`Error in script2: ${stderr}`);
    // }
    console.log(`Second script output: ${stdout}`);
    return stdout.trim();
  } catch (error) {
    console.error(`Failed to run script2: ${error}`);
    throw error;
  }
}

async function runPythonScript(): Promise<any> {
  try {
    // 1. Run second script in parallel (unaffected by other scripts)
    const secondScriptResult = await runSecondScript();

    // 2. Run the first and third scripts sequentially
    const firstScriptResult = await runFirstScript();
    const thirdScriptResult = await runThirdScript(firstScriptResult);

    console.log('All scripts executed successfully');

    // Handle stdout and stderr
    console.log('Script 1 output:', firstScriptResult);
    //console.error('Script 1 error:', result1.stderr);

    // Split output by lines
    const lines = secondScriptResult.split('\n');
    // Delete the first two lines
    const remainingLines = lines.slice(2);
    // Reassemble remaining lines into a string
    const newOutput = remainingLines.join('\n');
    console.log('Script 2 output:', newOutput);
    // console.error('Script 2 error:', result2.stderr);

    console.log('Script 3 output:', thirdScriptResult);
    //console.error('Script 3 error:', result3.stderr);

    return `${firstScriptResult}---${newOutput}---${thirdScriptResult}`;
  } catch (error) {
    console.error('Error executing Python scripts:', error);
    throw error;
  }
}
