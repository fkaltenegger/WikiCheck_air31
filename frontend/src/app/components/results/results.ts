import { Component, inject, signal} from '@angular/core';
import { catchError, Observable } from 'rxjs';
import { CheckService } from '../../services/check';
import { ResultItem } from '../../models/result.type';

@Component({
  selector: 'app-results',
  imports: [],
  templateUrl: './results.html',
  styleUrl: './results.css',
})
export class Results {

  constructor(public checkService: CheckService){
  }
}
