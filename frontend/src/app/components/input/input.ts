import { Component, inject, signal} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CheckService } from '../../services/check';


@Component({
  selector: 'app-input',
  imports: [CommonModule, FormsModule],
  templateUrl: './input.html',
  styleUrl: './input.css',
})
export class Input {
  
  query = signal('');
  methods = signal<Array<string>>(['TF-IDF', 'MBert', 'SBert']);
  method = signal<string>(this.methods()[0]);

  
  constructor(private checkService: CheckService) {}

  checkFact(){
    console.log(this.method(), this.query());
    this.checkService.checkFactApi(this.method(), this.query());
  }
}
