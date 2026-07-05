from services.retrieval import search_cars
from services.session_memory import (
    save_last_results,
    resolve_car_reference,
    get_selected_car,
    save_selected_car,
)


session_id = "test_session_1"

# Step 1: Pretend the user searched for Mercedes cars
cars = search_cars(make="mercedes-benz", limit=5)
save_last_results(session_id, cars)

print("\nSaved last search results:")
for index, car in enumerate(cars, start=1):
    print(f"{index}. {car['year']} {car['make']} {car['model']} - Listing ID {car['listing_id']}")

# Step 2: User says "Tell me about the first one"
first_car = resolve_car_reference(session_id, "Tell me about the first one")

print("\nResolved 'first one' to:")
print(first_car["title"])

# Step 3: Save it as selected car
save_selected_car(session_id, first_car)

# Step 4: User says "Does it mention warranty?"
selected_car = get_selected_car(session_id)

print("\nResolved 'it' to:")
print(selected_car["title"])

text = (selected_car["title"] + " " + selected_car["description"]).lower()

if "warranty" in text:
    print("\nYes, this selected listing mentions warranty.")
else:
    print("\nNo, this selected listing does not mention warranty.")