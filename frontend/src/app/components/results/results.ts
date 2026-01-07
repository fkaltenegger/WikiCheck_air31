import { Component, inject, signal} from '@angular/core';
import { catchError, Observable } from 'rxjs';
import { CheckService } from '../../services/check-service';
import { ResultItem } from '../../models/result.type';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-results',
  imports: [CommonModule],
  templateUrl: './results.html',
  styleUrl: './results.css',
})
export class Results {

  constructor(public checkService: CheckService){}

  getEvalBackground(evalValue: string): string {
  switch (evalValue) {
    case 'SUPPORTS':
      return 'bg-green-500';
    case 'NOT MENTIONED':
      return 'bg-gray-500';
    case 'CONTRADICTS':
      return 'bg-red-500';
    default:
      return 'bg-gray-400';
  }
}
}
