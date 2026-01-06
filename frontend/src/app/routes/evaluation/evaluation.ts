import { Component, OnInit } from '@angular/core';
import { EvaluationService } from '../../services/evaluation';

@Component({
  selector: 'app-evaluation',
  imports: [],
  templateUrl: './evaluation.html',
  styleUrl: './evaluation.css',
})
export class Evaluation implements OnInit {

  constructor(public evaluationService: EvaluationService){}

  ngOnInit(): void {
    this.getEvaluation();
  }

  getEvaluation() {
    this.evaluationService.getEvaluation();
    console.log(this.evaluationService.data());
  }

}
