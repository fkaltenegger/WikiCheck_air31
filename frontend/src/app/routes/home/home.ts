import { Component, inject, signal} from '@angular/core';
import { catchError } from 'rxjs';
import { RouterLink } from "@angular/router";
import { Results } from "../../components/results/results";

@Component({
  selector: 'app-home',
  imports: [RouterLink],
  templateUrl: './home.html',
  styleUrl: './home.css',
})
export class Home{

}
